//! App errors
//!
//! This crate provides an error type, [`AppError`], that is ideal for usage in apps.
//!
//! It is [`Send`], [`Sync`], `'static`, and importantly cheaply [`Clone`]-able.
//!
//! It is also able to store multiple errors at once and provide pretty-printing of all
//! of these errors.
//!
//! The inner representation is currently just `Arc<(String, Option<AppError>) | Box<[AppError]>>`.

// Features
#![feature(error_reporter, decl_macro, try_trait_v2, extend_one, let_chains)]

// Modules
mod multiple;
mod pretty;

// Exports
pub use self::{multiple::AllErrs, pretty::PrettyDisplay};

// Imports
use {
	core::mem,
	std::{
		error::Error as StdError,
		fmt,
		hash::{Hash, Hasher},
		sync::Arc,
	},
};

/// Inner
enum Inner<D> {
	/// Single error
	Single {
		/// Message
		msg: String,

		/// Source
		source: Option<AppError<D>>,

		/// User data
		data: D,
	},

	/// Multiple errors
	Multiple(Box<[AppError<D>]>),
}

impl<D> StdError for Inner<D>
where
	D: fmt::Debug + 'static,
{
	fn source(&self) -> Option<&(dyn StdError + 'static)> {
		match self {
			Inner::Single { source, .. } => source.as_ref().map(AppError::as_std_error),
			// For standard errors, just use the first source.
			Inner::Multiple(errs) => errs.first().map(AppError::as_std_error),
		}
	}
}

impl<D> PartialEq for Inner<D>
where
	D: PartialEq,
{
	fn eq(&self, other: &Self) -> bool {
		match (self, other) {
			(
				Self::Single {
					msg: lhs_msg,
					source: lhs_source,
					data: lhs_data,
				},
				Self::Single {
					msg: rhs_msg,
					source: rhs_source,
					data: rhs_data,
				},
			) => lhs_msg == rhs_msg && lhs_source == rhs_source && lhs_data == rhs_data,
			(Self::Multiple(lhs), Self::Multiple(rhs)) => lhs == rhs,
			_ => false,
		}
	}
}

impl<D> Hash for Inner<D>
where
	D: Hash,
{
	fn hash<H: Hasher>(&self, state: &mut H) {
		mem::discriminant(self).hash(state);
		match self {
			Inner::Single { msg, source, data } => {
				msg.hash(state);
				source.hash(state);
				data.hash(state);
			},
			Inner::Multiple(errs) => errs.hash(state),
		}
	}
}

impl<D> fmt::Display for Inner<D> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Inner::Single { msg, .. } => msg.fmt(f),
			Inner::Multiple(errs) => write!(f, "Multiple errors ({})", errs.len()),
		}
	}
}

impl<D> fmt::Debug for Inner<D>
where
	D: fmt::Debug + 'static,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match f.alternate() {
			// With `:#?`, use a normal debug
			true => match self {
				Inner::Single { msg, source, data } => f
					.debug_struct("AppError")
					.field("msg", msg)
					.field("source", source)
					.field("data", data)
					.finish(),
				Inner::Multiple(errs) => f.debug_list().entries(errs).finish(),
			},

			// Otherwise, pretty print it
			false => std::error::Report::new(self).pretty(true).fmt(f),
		}
	}
}

/// A reference-counted untyped error that can be created from any error type.
///
/// Named `AppError` as it's mostly useful in apps that don't care about the errors
/// specifically, and instead only care to show them to users.
pub struct AppError<D = ()> {
	/// Inner
	inner: Arc<Inner<D>>,
}

impl<D> AppError<D> {
	/// Creates a new app error from an error
	pub fn new<E>(err: &E) -> Self
	where
		E: ?Sized + StdError,
		D: Default,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg:    err.to_string(),
				source: err.source().map(Self::new),
				data:   D::default(),
			}),
		}
	}

	/// Creates a new app error from an error and data.
	///
	/// `data` will be applied to all sources of `err`
	pub fn new_with_data<E>(err: &E, data: D) -> Self
	where
		E: ?Sized + StdError,
		D: Clone,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg: err.to_string(),
				source: err.source().map(|source| Self::new_with_data(source, data.clone())),
				data,
			}),
		}
	}

	/// Creates a new app error from a message
	pub fn msg<M>(msg: M) -> Self
	where
		M: fmt::Display,
		D: Default,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg:    msg.to_string(),
				source: None,
				data:   D::default(),
			}),
		}
	}

	/// Creates a new app error from a message
	pub fn msg_with_data<M>(msg: M, data: D) -> Self
	where
		M: fmt::Display,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg: msg.to_string(),
				source: None,
				data,
			}),
		}
	}

	/// Adds context to this error
	pub fn context<M>(&self, msg: M) -> Self
	where
		M: fmt::Display,
		D: Default,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg:    msg.to_string(),
				source: Some(self.clone()),
				data:   D::default(),
			}),
		}
	}

	/// Adds context to this error
	pub fn context_with_data<M>(&self, msg: M, data: D) -> Self
	where
		M: fmt::Display,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg: msg.to_string(),
				source: Some(self.clone()),
				data,
			}),
		}
	}

	/// Creates a new app error from multiple errors
	pub fn from_multiple<Errs>(errs: Errs) -> Self
	where
		Errs: IntoIterator<Item = Self>,
	{
		Self {
			inner: Arc::new(Inner::Multiple(errs.into_iter().collect())),
		}
	}

	/// Creates a new app error from multiple standard errors
	pub fn from_multiple_std<'a, Errs, E>(errs: Errs) -> Self
	where
		Errs: IntoIterator<Item = &'a E>,
		E: ?Sized + StdError + 'a,
		D: Default,
	{
		Self {
			inner: Arc::new(Inner::Multiple(errs.into_iter().map(Self::new).collect())),
		}
	}

	/// Returns this type as a [`std::error::Error`]
	pub fn as_std_error(&self) -> &(dyn StdError + 'static)
	where
		D: fmt::Debug + 'static,
	{
		&self.inner
	}

	/// Converts this type as into a [`std::error::Error`]
	pub fn into_std_error(self) -> Arc<dyn StdError + Send + Sync + 'static>
	where
		D: fmt::Debug + Send + Sync + 'static,
	{
		self.inner as Arc<_>
	}

	/// Returns an object that can be used for a pretty display of this error
	#[must_use]
	pub fn pretty(&self) -> PrettyDisplay<'_, D> {
		PrettyDisplay::new(self)
	}
}

impl<D> Clone for AppError<D> {
	fn clone(&self) -> Self {
		Self {
			inner: Arc::clone(&self.inner),
		}
	}
}


impl<E, D> From<E> for AppError<D>
where
	E: StdError,
	D: Default,
{
	fn from(err: E) -> Self {
		Self::new(&err)
	}
}

impl<D> PartialEq for AppError<D>
where
	D: PartialEq,
{
	fn eq(&self, other: &Self) -> bool {
		// If we're the same Arc, we're the same error
		if Arc::ptr_eq(&self.inner, &other.inner) {
			return true;
		}

		// Otherwise, perform a deep comparison
		self.inner == other.inner
	}
}

impl<D> Eq for AppError<D> where D: Eq {}

impl<D> Hash for AppError<D>
where
	D: Hash,
{
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.inner.hash(state);
	}
}

impl<D> fmt::Display for AppError<D> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.inner.fmt(f)
	}
}

impl<D> fmt::Debug for AppError<D>
where
	D: fmt::Debug + 'static,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.inner.fmt(f)
	}
}

/// Context for `Result`-like types
pub trait Context<D> {
	type Output;

	/// Adds context to this result, if it's an error
	fn context<M>(self, msg: M) -> Self::Output
	where
		M: fmt::Display;

	/// Adds context to this result lazily, if it's an error
	fn with_context<F, M>(self, with_msg: F) -> Self::Output
	where
		F: FnOnce() -> M,
		M: fmt::Display;
}

impl<T, E, D> Context<D> for Result<T, E>
where
	E: StdError,
	D: Default,
{
	type Output = Result<T, AppError<D>>;

	fn context<M>(self, msg: M) -> Self::Output
	where
		M: fmt::Display,
	{
		self.map_err(|err| AppError::new(&err).context(msg))
	}

	fn with_context<F, M>(self, with_msg: F) -> Self::Output
	where
		F: FnOnce() -> M,
		M: fmt::Display,
	{
		self.map_err(|err| AppError::new(&err).context(with_msg()))
	}
}

impl<T, D> Context<D> for Result<T, AppError<D>>
where
	D: Default,
{
	type Output = Result<T, AppError<D>>;

	fn context<M>(self, msg: M) -> Self::Output
	where
		M: fmt::Display,
	{
		self.map_err(|err| err.context(msg))
	}

	fn with_context<F, M>(self, with_msg: F) -> Self::Output
	where
		F: FnOnce() -> M,
		M: fmt::Display,
	{
		self.map_err(|err| err.context(with_msg()))
	}
}

impl<T, D> Context<D> for Option<T>
where
	D: Default,
{
	type Output = Result<T, AppError<D>>;

	fn context<M>(self, msg: M) -> Self::Output
	where
		M: fmt::Display,
	{
		self.ok_or_else(|| AppError::msg(msg))
	}

	fn with_context<F, M>(self, with_msg: F) -> Self::Output
	where
		F: FnOnce() -> M,
		M: fmt::Display,
	{
		self.ok_or_else(|| AppError::msg(with_msg()))
	}
}

/// A macro that formats and creates an [`AppError`]
pub macro app_error {
	($msg:expr $(,)?) => {
		$crate::AppError::msg( format!($msg) )
	},

	($fmt:expr, $($arg:expr),* $(,)?) => {
		$crate::AppError::msg( format!($fmt, $($arg,)*) )
	},
}

/// A macro that returns an error
pub macro bail {
	($msg:expr $(,)?) => {
		do yeet $crate::app_error!($msg);
	},

	($fmt:expr, $($arg:expr),* $(,)?) => {
		do yeet $crate::app_error!($fmt, $($arg),*);
	},
}

/// A macro that returns an error if a condition is false
pub macro ensure {
	($cond:expr, $msg:expr $(,)?) => {
		if !$cond {
			do yeet $crate::app_error!($msg);
		}
	},

	($cond:expr, $fmt:expr, $($arg:expr),* $(,)?) => {
		if !$cond {
			do yeet $crate::app_error!($fmt, $($arg),*);
		}
	},
}
