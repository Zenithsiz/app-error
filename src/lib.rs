//! App errors
//!
//! This crate provides an error type, [`AppError`], that is intended for usage in apps.
//!
//! It is [`Send`], [`Sync`], `'static`, and importantly cheaply [`Clone`]-able.
//!
//! To achieve this, it serializes every error it receives without owning it, meaning that
//! you also can't retrieve the error later by downcasting it.
//!
//! It is also able to store multiple errors at once and provide pretty-printing of all
//! of these them.
//!
//! The inner representation is similar to `AppError = (String, Option<AppError>) | Vec<AppError>`.

// Features
#![feature(
	decl_macro,
	try_trait_v2,
	extend_one,
	debug_closure_helpers,
	try_blocks,
	never_type,
	unwrap_infallible
)]
#![cfg_attr(test, feature(assert_matches, coverage_attribute, yeet_expr))]

// Modules
mod multiple;
mod pretty;

// Exports
pub use self::{multiple::AllErrs, pretty::PrettyDisplay};

// Imports
use {
	core::{mem, slice},
	std::{
		borrow::Cow,
		error::Error as StdError,
		fmt,
		hash::{Hash, Hasher},
		sync::Arc,
	},
};

/// Inner representation.
#[derive(Clone)]
enum Inner<D> {
	/// Single error
	Single {
		/// Message
		msg: Cow<'static, str>,

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
			Self::Single { source, .. } => source.as_ref().map(AppError::as_std_error),
			// For standard errors, just use the first source.
			Self::Multiple(errs) => errs.first().map(AppError::as_std_error),
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
			Self::Single { msg, source, data } => {
				msg.hash(state);
				source.hash(state);
				data.hash(state);
			},
			Self::Multiple(errs) => errs.hash(state),
		}
	}
}

impl<D> fmt::Display for Inner<D> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Single { msg, .. } => msg.fmt(f),
			Self::Multiple(errs) => write!(f, "Multiple errors ({})", errs.len()),
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
				Self::Single { msg, source, data } => f
					.debug_struct("AppError")
					.field("msg", msg)
					.field("source", source)
					.field("data", data)
					.finish(),
				Self::Multiple(errs) => f.debug_list().entries(errs).finish(),
			},

			// Otherwise, pretty print it
			false => write!(f, "{}", PrettyDisplay::new(self)),
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
				msg:    err.to_string().into(),
				source: err.source().map(Self::new),
				data:   D::default(),
			}),
		}
	}

	/// Creates a new app error from a message
	#[must_use]
	pub fn msg(msg: &'static str) -> Self
	where
		D: Default,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg:    msg.into(),
				source: None,
				data:   D::default(),
			}),
		}
	}

	/// Creates a new app error from a formattable message
	pub fn fmt<M>(msg: M) -> Self
	where
		M: fmt::Display,
		D: Default,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg:    msg.to_string().into(),
				source: None,
				data:   D::default(),
			}),
		}
	}

	/// Adds context to this error
	#[must_use = "Creates a new error with context, without modifying the existing one"]
	pub fn context(&self, msg: &'static str) -> Self
	where
		D: Default,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg:    msg.into(),
				source: Some(self.clone()),
				data:   D::default(),
			}),
		}
	}

	/// Adds context to this error
	#[must_use = "Creates a new error with context, without modifying the existing one"]
	pub fn with_context<F, M>(&self, with_msg: F) -> Self
	where
		F: FnOnce() -> M,
		M: fmt::Display,
		D: Default,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg:    with_msg().to_string().into(),
				source: Some(self.clone()),
				data:   D::default(),
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

	/// Flattens all neighbor multiple errors into a single one.
	#[must_use]
	pub fn flatten(self) -> Self
	where
		D: Clone,
	{
		/// Gathers all multiple errors from `err` into `flattened_errs`, recursively
		fn flatten_into<D: Clone>(err: AppError<D>, flattened_errs: &mut Vec<AppError<D>>) {
			let err = err.flatten();
			match &*err.inner {
				Inner::Single { .. } => flattened_errs.push(err),
				Inner::Multiple(errs) =>
					for err in errs {
						flatten_into(err.clone(), flattened_errs);
					},
			}
		}

		fn flatten_inner<D: Clone>(err: AppError<D>) -> Option<AppError<D>> {
			match Arc::unwrap_or_clone(err.inner) {
				// If we're a single error, recurse
				Inner::Single { msg, source, data } => Some(AppError {
					inner: Arc::new(Inner::Single {
						msg,
						source: source.and_then(flatten_inner),
						data,
					}),
				}),

				// Otherwise, flatten all errors
				Inner::Multiple(errs) => {
					let mut flattened_errs = vec![];
					for err in errs {
						flatten_into(err, &mut flattened_errs);
					}

					match <[_; 0]>::try_from(flattened_errs) {
						Ok([]) => None,
						Err(flattened_errs) => match <[_; 1]>::try_from(flattened_errs) {
							Ok([err]) => Some(err),
							Err(flattened_errs) => Some(AppError::from_multiple(flattened_errs)),
						},
					}
				},
			}
		}

		flatten_inner(self).unwrap_or_else(|| Self::from_multiple([]))
	}

	/// Returns this type as a [`std::error::Error`]
	#[must_use]
	pub fn as_std_error(&self) -> &(dyn StdError + 'static)
	where
		D: fmt::Debug + 'static,
	{
		&self.inner
	}

	/// Converts this type as into a [`std::error::Error`]
	#[must_use]
	pub fn into_std_error(self) -> Arc<dyn StdError + Send + Sync + 'static>
	where
		D: fmt::Debug + Send + Sync + 'static,
	{
		self.inner as Arc<_>
	}

	/// Returns an object that can be used for a pretty display of this error
	#[must_use]
	pub fn pretty(&self) -> PrettyDisplay<'_, D> {
		PrettyDisplay::new(&self.inner)
	}

	/// Returns an iterator over all data in this error, recursively.
	///
	/// # Order
	/// No order is guaranteed and it may change at any time.
	#[must_use]
	pub fn data_iter(&self) -> DataIter<'_, D> {
		DataIter {
			errs:   vec![self],
			cur_it: None,
		}
	}
}

impl<D> AppError<D> {
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
				msg: err.to_string().into(),
				source: err.source().map(|source| Self::new_with_data(source, data.clone())),
				data,
			}),
		}
	}

	/// Creates a new app error from a message
	pub fn msg_with_data(msg: &'static str, data: D) -> Self {
		Self {
			inner: Arc::new(Inner::Single {
				msg: msg.into(),
				source: None,
				data,
			}),
		}
	}

	/// Creates a new app error from a formatted message
	pub fn fmt_with_data<M>(msg: M, data: D) -> Self
	where
		M: fmt::Display,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg: msg.to_string().into(),
				source: None,
				data,
			}),
		}
	}

	/// Adds context to this error
	#[must_use = "Creates a new error with context, without modifying the existing one"]
	pub fn context_with_data(&self, msg: &'static str, data: D) -> Self {
		Self {
			inner: Arc::new(Inner::Single {
				msg: msg.to_string().into(),
				source: Some(self.clone()),
				data,
			}),
		}
	}

	/// Adds context to this error
	#[must_use = "Creates a new error with context, without modifying the existing one"]
	pub fn with_context_with_data<F, M>(&self, with_msg: F, data: D) -> Self
	where
		F: Fn() -> M,
		M: fmt::Display,
	{
		Self {
			inner: Arc::new(Inner::Single {
				msg: with_msg().to_string().into(),
				source: Some(self.clone()),
				data,
			}),
		}
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

/// Data iterator
#[derive(Clone, Debug)]
pub struct DataIter<'a, D: 'static> {
	/// Error stack
	errs: Vec<&'a AppError<D>>,

	/// Current multiple iter
	cur_it: Option<slice::Iter<'a, AppError<D>>>,
}

impl<'a, D: 'static> Iterator for DataIter<'a, D> {
	type Item = (&'a AppError<D>, &'a D);

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			// Get the next error to check
			let next_err = 'next_err: {
				// If we're in the middle of an iterator, and we got a new one, then use it
				if let Some(cur_it) = &mut self.cur_it {
					match cur_it.next() {
						Some(err) => break 'next_err err,
						// Note: Remember to remove the iterator once exhausted
						None => self.cur_it = None,
					}
				}

				// Otherwise, use our next error from the stack
				match self.errs.pop() {
					Some(err) => break 'next_err err,
					// If we have no more errors in the stack, we're done
					None => return None,
				}
			};

			// Then check if we can use it
			match &*next_err.inner {
				// If it's just a single error, yield the data and stack it's
				// source onto the stack for next.
				Inner::Single { source, data, .. } => {
					if let Some(source) = source {
						self.errs.push(source);
					}

					break Some((next_err, data));
				},

				// Otherwise, if it's multiple
				Inner::Multiple(errs) => match &self.cur_it {
					// If we're already iterating, save it for later
					Some(_) => self.errs.push(next_err),

					// Otherwise, use it as our current iterator
					None => self.cur_it = Some(errs.iter()),
				},
			}
		}
	}
}

/// Context for `Result`-like types
pub trait Context<D> {
	type Output;

	/// Adds context to this result, if it's an error
	fn context(self, msg: &'static str) -> Self::Output;

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

	fn context(self, msg: &'static str) -> Self::Output {
		self.map_err(|err| AppError::new(&err).context(msg))
	}

	fn with_context<F, M>(self, with_msg: F) -> Self::Output
	where
		F: FnOnce() -> M,
		M: fmt::Display,
	{
		self.map_err(|err| AppError::new(&err).with_context(with_msg))
	}
}

impl<T, D> Context<D> for Result<T, AppError<D>>
where
	D: Default,
{
	type Output = Self;

	fn context(self, msg: &'static str) -> Self::Output {
		self.map_err(|err| err.context(msg))
	}

	fn with_context<F, M>(self, with_msg: F) -> Self::Output
	where
		F: FnOnce() -> M,
		M: fmt::Display,
	{
		self.map_err(|err| err.with_context(with_msg))
	}
}

impl<T, D> Context<D> for Result<T, &AppError<D>>
where
	D: Default,
{
	type Output = Result<T, AppError<D>>;

	fn context(self, msg: &'static str) -> Self::Output {
		self.map_err(|err| err.context(msg))
	}

	fn with_context<F, M>(self, with_msg: F) -> Self::Output
	where
		F: FnOnce() -> M,
		M: fmt::Display,
	{
		self.map_err(|err| err.with_context(with_msg))
	}
}

impl<T, D> Context<D> for Result<T, &mut AppError<D>>
where
	D: Default,
{
	type Output = Result<T, AppError<D>>;

	fn context(self, msg: &'static str) -> Self::Output {
		self.map_err(|err| err.context(msg))
	}

	fn with_context<F, M>(self, with_msg: F) -> Self::Output
	where
		F: FnOnce() -> M,
		M: fmt::Display,
	{
		self.map_err(|err| err.with_context(with_msg))
	}
}

impl<T, D> Context<D> for Option<T>
where
	D: Default,
{
	type Output = Result<T, AppError<D>>;

	fn context(self, msg: &'static str) -> Self::Output {
		self.ok_or_else(|| AppError::msg(msg))
	}

	fn with_context<F, M>(self, with_msg: F) -> Self::Output
	where
		F: FnOnce() -> M,
		M: fmt::Display,
	{
		self.ok_or_else(|| AppError::fmt(with_msg()))
	}
}

/// A macro that formats and creates an [`AppError`]
pub macro app_error {
	($msg:literal $(,)?) => {
		// TODO: Check if it's a static string as compile time?
		match format_args!($msg) {
			msg => match msg.as_str() {
				Some(msg) => $crate::AppError::msg(msg),
				None => $crate::AppError::fmt( ::std::fmt::format(msg) )
			}
		}

	},

	($fmt:literal, $($arg:expr),* $(,)?) => {
		$crate::AppError::fmt( format!($fmt, $($arg,)*) )
	},
}

/// A macro that returns an error
pub macro bail {
	($msg:literal $(,)?) => {
		do yeet $crate::app_error!($msg)
	},

	($fmt:literal, $($arg:expr),* $(,)?) => {
		do yeet $crate::app_error!($fmt, $($arg),*)
	},
}

/// A macro that returns an error if a condition is false
pub macro ensure {
	($cond:expr, $msg:literal $(,)?) => {
		if !$cond {
			do yeet $crate::app_error!($msg);
		}
	},

	($cond:expr, $fmt:literal, $($arg:expr),* $(,)?) => {
		if !$cond {
			do yeet $crate::app_error!($fmt, $($arg),*);
		}
	},
}

#[cfg(test)]
#[cfg_attr(test, coverage(off))]
mod test {
	use {
		super::*,
		std::{assert_matches::assert_matches, collections::HashSet},
	};

	/// Error implementing `StdError` for testing.
	#[derive(Clone, Debug)]
	struct StdE {
		msg:   &'static str,
		inner: Option<Box<Self>>,
	}
	impl StdError for StdE {
		fn source(&self) -> Option<&(dyn StdError + 'static)> {
			self.inner.as_deref().map(|err| err as &dyn StdError)
		}
	}
	impl fmt::Display for StdE {
		fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			self.msg.fmt(f)
		}
	}

	#[test]
	fn std_error_source() {
		let err = AppError::<()>::msg("A");
		assert_matches!(err.as_std_error().source(), None);
		assert_matches!(err.into_std_error().source(), None);

		let err = AppError::<()>::msg("A").context("B");
		assert_matches!(err.as_std_error().source(), Some(err) if err.to_string() == "A");
		assert_matches!(err.into_std_error().source(), Some(err) if err.to_string() == "A");

		let err = AppError::<()>::from_multiple([AppError::msg("A"), AppError::msg("B")]);
		assert_matches!(err.as_std_error().source(), Some(err) if err.to_string() == "A");
		assert_matches!(err.into_std_error().source(), Some(err) if err.to_string() == "A");
	}

	#[test]
	fn std_error_fmt() {
		let err = AppError::<()>::msg("A");
		assert_eq!(err.as_std_error().to_string(), "A");
		assert_eq!(err.into_std_error().to_string(), "A");

		let err = AppError::<()>::msg("A").context("B");
		assert_eq!(err.as_std_error().to_string(), "B");
		assert_eq!(err.into_std_error().to_string(), "B");

		let err = AppError::<()>::from_multiple([AppError::msg("A"), AppError::msg("B")]);
		assert_eq!(err.as_std_error().to_string(), "Multiple errors (2)");
		assert_eq!(err.into_std_error().to_string(), "Multiple errors (2)");
	}

	#[test]
	fn from_std_error() {
		let std_err = StdE {
			msg:   "A",
			inner: Some(Box::new(StdE {
				msg:   "B",
				inner: None,
			})),
		};

		let found = AppError::<()>::new(&std_err);
		assert_eq!(found, AppError::from(std_err));
		let expected = AppError::msg("B").context("A");
		assert!(
			found == expected,
			"Converted error was wrong.\nExpected: {}\nFound: {}",
			expected.pretty(),
			found.pretty()
		);
	}

	#[test]
	fn from_multiple_std_error() {
		let std_errs = [
			StdE {
				msg:   "A",
				inner: Some(Box::new(StdE {
					msg:   "B",
					inner: None,
				})),
			},
			StdE {
				msg:   "C",
				inner: Some(Box::new(StdE {
					msg:   "D",
					inner: None,
				})),
			},
		];

		let found = AppError::<()>::from_multiple_std(&std_errs);
		let expected =
			AppError::<()>::from_multiple([AppError::msg("B").context("A"), AppError::msg("D").context("C")]);
		assert!(
			found == expected,
			"Converted error was wrong.\nExpected: {}\nFound: {}",
			expected.pretty(),
			found.pretty()
		);
	}

	#[test]
	fn eq() {
		let err_a1 = AppError::<()>::msg("A");
		assert_eq!(err_a1, err_a1);
		assert_eq!(err_a1.clone(), err_a1);

		let err_a2 = AppError::<()>::msg("A");
		assert_eq!(err_a1, err_a2);

		let err_b = AppError::<()>::msg("B");
		assert_ne!(err_a1, err_b);
	}

	#[test]
	fn eq_context() {
		let err1 = AppError::<()>::msg("A").context("B");
		assert_eq!(err1, err1);

		let err2 = AppError::<()>::msg("A").context("B");
		assert_eq!(err1, err2);

		let err3 = AppError::<()>::msg("A").context("C");
		assert_ne!(err1, err3);

		let err4 = AppError::<()>::msg("B").context("C");
		assert_ne!(err3, err4);
	}

	#[test]
	fn eq_multiple() {
		let err_multiple1 = AppError::<()>::from_multiple([AppError::msg("A"), AppError::msg("B")]);
		assert_eq!(err_multiple1, err_multiple1);
		assert_eq!(err_multiple1.clone(), err_multiple1);

		let err_single = AppError::<()>::msg("A");
		assert_ne!(err_multiple1, err_single);

		let err_multiple2 = AppError::<()>::from_multiple([AppError::msg("A"), AppError::msg("B")]);
		assert_eq!(err_multiple1, err_multiple2);

		let err_multiple3 = AppError::<()>::from_multiple([AppError::msg("B"), AppError::msg("A")]);
		assert_ne!(err_multiple1, err_multiple3);
	}

	#[test]
	fn eq_data() {
		#[derive(PartialEq, Debug)]
		struct D(usize);

		let err1 = AppError::msg_with_data("A", D(0));
		assert_eq!(err1, err1);

		let err2 = AppError::msg_with_data("A", D(1));
		assert_ne!(err1, err2);
	}

	#[test]
	fn hash() {
		let mut errs = HashSet::new();
		let err = AppError::<()>::msg("A");
		assert!(errs.insert(err.clone()));
		assert!(!errs.insert(err));
		assert!(!errs.insert(AppError::<()>::msg("A")));

		let err_multiple = AppError::<()>::from_multiple([AppError::msg("A"), AppError::msg("B")]);
		assert!(errs.insert(err_multiple.clone()));
		assert!(!errs.insert(err_multiple));
		assert!(!errs.insert(AppError::<()>::from_multiple([AppError::msg("A"), AppError::msg("B")])));
	}

	#[test]
	fn fmt_debug() {
		let err = AppError::<()>::msg("A").context("B");
		assert_eq!(format!("{err:?}"), "B\n└─A");
		assert_eq!(
			format!("{err:#?}"),
			r#"AppError {
    msg: "B",
    source: Some(
        AppError {
            msg: "A",
            source: None,
            data: (),
        },
    ),
    data: (),
}"#
		);

		let err_multiple = AppError::<()>::from_multiple([AppError::msg("A"), AppError::msg("B")]);
		assert_eq!(format!("{err_multiple:?}"), "Multiple errors:\n├─A\n└─B");
		assert_eq!(
			format!("{err_multiple:#?}"),
			r#"[
    AppError {
        msg: "A",
        source: None,
        data: (),
    },
    AppError {
        msg: "B",
        source: None,
        data: (),
    },
]"#
		);
	}

	#[test]
	fn fmt() {
		assert_eq!(AppError::<()>::fmt("ABC").to_string(), "ABC");
	}

	#[test]
	fn with_context() {
		assert_eq!(
			AppError::<()>::msg("A").context("B"),
			AppError::<()>::msg("A").with_context(|| "B")
		);
	}

	#[test]
	fn ext_traits() {
		let std_err = StdE {
			msg:   "B",
			inner: Some(Box::new(StdE {
				msg:   "C",
				inner: None,
			})),
		};

		assert_eq!(
			Err::<(), StdE>(std_err.clone()).context("A"),
			Err(AppError::<()>::msg("C").context("B").context("A"))
		);
		assert_eq!(
			Err::<(), StdE>(std_err).with_context(|| "A"),
			Err(AppError::<()>::msg("C").context("B").context("A"))
		);

		assert_eq!(
			Err::<(), AppError>(AppError::msg("C").context("B")).context("A"),
			Err(AppError::<()>::msg("C").context("B").context("A"))
		);
		assert_eq!(
			Err::<(), AppError>(AppError::msg("C").context("B")).with_context(|| "A"),
			Err(AppError::<()>::msg("C").context("B").context("A"))
		);

		assert_eq!(None::<()>.context("A"), Err(AppError::<()>::msg("A")));
		assert_eq!(None::<()>.with_context(|| "A"), Err(AppError::<()>::msg("A")));
	}

	#[test]
	fn data_from_std() {
		#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug)]
		struct D(usize);

		let std_err = StdE {
			msg:   "A",
			inner: Some(Box::new(StdE {
				msg:   "B",
				inner: None,
			})),
		};

		let err = AppError::<D>::new_with_data(&std_err, D(5));
		assert_eq!(
			err.data_iter().map(|(_, &d)| d).collect::<HashSet<_>>(),
			[D(5), D(5)].into()
		);
		assert_eq!(
			err,
			AppError::<D>::msg_with_data("B", D(5)).with_context_with_data(|| "A", D(5))
		);
	}

	#[test]
	fn data() {
		#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug)]
		struct D(usize);

		let err = AppError::<D>::fmt_with_data("B", D(4)).context_with_data("A", D(5));
		assert_eq!(
			err.data_iter().map(|(_, &d)| d).collect::<HashSet<_>>(),
			[D(5), D(4)].into()
		);
	}

	#[test]
	fn data_multiple() {
		#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug)]
		struct D(usize);

		let err = AppError::from_multiple([AppError::msg_with_data("A", D(1)), AppError::msg_with_data("B", D(2))]);
		assert_eq!(
			err.data_iter().map(|(_, &d)| d).collect::<HashSet<_>>(),
			[D(1), D(2)].into()
		);
	}

	#[test]
	fn data_empty() {
		let err = AppError::<!>::from_multiple([]);
		assert_eq!(err.data_iter().map(|(_, &d)| d).collect::<Vec<_>>(), []);
	}

	#[test]
	fn data_complex() {
		#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug)]
		struct D(usize);

		let err = AppError::<D>::from_multiple([
			AppError::msg_with_data("B", D(1)).context_with_data("A", D(0)),
			AppError::from_multiple([
				AppError::msg_with_data("C", D(2)),
				AppError::msg_with_data("D", D(3)),
				AppError::from_multiple([AppError::msg_with_data("E", D(4)), AppError::msg_with_data("F", D(5))]),
				AppError::from_multiple([AppError::msg_with_data("G", D(6)), AppError::msg_with_data("H", D(7))]),
			]),
			AppError::from_multiple([AppError::msg_with_data("I", D(8))]),
			AppError::msg_with_data("J", D(9)),
		]);
		assert_eq!(
			err.data_iter().map(|(_, &d)| d).collect::<HashSet<_>>(),
			(0..=9).map(D).collect()
		);
	}

	#[test]
	fn pretty() {
		let err = AppError::<()>::msg("A").context("B\nC").context("D");
		assert_eq!(
			format!("{:#?}", err.pretty()),
			r#"PrettyDisplay {
    root: AppError {
        msg: "D",
        source: Some(
            AppError {
                msg: "B\nC",
                source: Some(
                    AppError {
                        msg: "A",
                        source: None,
                        data: (),
                    },
                ),
                data: (),
            },
        ),
        data: (),
    },
    ignore_err: None,
}"#
		);
		assert_eq!(
			err.pretty().to_string(),
			r"D
└─B
  C
  └─A"
		);
		assert_eq!(err.pretty().to_string(), format!("{:?}", err.pretty()));

		let err_multiple = AppError::<()>::from_multiple([AppError::msg("A"), AppError::msg("B")]);
		assert_eq!(
			err_multiple.pretty().to_string(),
			r"Multiple errors:
├─A
└─B"
		);

		let err_multiple_deep = AppError::<()>::from_multiple([
			AppError::from_multiple([AppError::msg("A\nA2"), AppError::msg("B\nB2")])
				.context("C\nC2")
				.context("D"),
			AppError::msg("E"),
		]);
		assert_eq!(
			err_multiple_deep.pretty().to_string(),
			r"Multiple errors:
├─D
│ └─C
│   C2
│   └─Multiple errors:
│     ├─A
│     │ A2
│     └─B
│       B2
└─E"
		);
	}

	#[test]
	fn pretty_ignore() {
		#[derive(Default)]
		struct D {
			ignore: bool,
		}

		let fmt_err = |err: &AppError<D>| err.pretty().with_ignore_err(|data| data.ignore).to_string();

		// TODO: This is not correct behavior, we shouldn't display the ignored errors just because there's no multiple
		let err = AppError::<D>::msg_with_data("A", D { ignore: true }).context("B");
		assert_eq!(fmt_err(&err), "B\n└─A");

		let err_multiple_deep = AppError::<D>::from_multiple([
			AppError::from_multiple([AppError::msg("A"), AppError::msg_with_data("B", D { ignore: true })])
				.context("C")
				.context("D"),
			AppError::msg_with_data("E", D { ignore: true }),
			AppError::msg_with_data("F", D { ignore: true }).context("G"),
			AppError::msg("H").context_with_data("I", D { ignore: true }),
			AppError::msg("J"),
		]);
		assert_eq!(
			fmt_err(&err_multiple_deep),
			r"Multiple errors:
├─D
│ └─C
│   └─Multiple errors:
│     ├─A
│     └─(1 ignored errors)
├─J
└─(3 ignored errors)"
		);
	}

	#[test]
	fn macros_static() {
		assert_eq!(app_error!("A"), AppError::<()>::msg("A"));

		#[expect(clippy::diverging_sub_expression)]
		let res: Result<(), AppError> = try { bail!("A") };
		assert_eq!(res, Err(AppError::<()>::msg("A")));

		let res: Result<(), AppError> = try { ensure!(true, "A") };
		assert_eq!(res, Ok(()));

		let res: Result<(), AppError> = try { ensure!(false, "A") };
		assert_eq!(res, Err(AppError::<()>::msg("A")));
	}

	#[test]
	fn macros_fmt() {
		let value = 5;
		assert_eq!(app_error!("A{value}"), AppError::<()>::msg("A5"));

		#[expect(clippy::diverging_sub_expression)]
		let res: Result<(), AppError> = try { bail!("A{value}") };
		assert_eq!(res, Err(AppError::<()>::msg("A5")));

		let res: Result<(), AppError> = try { ensure!(true, "A{value}") };
		assert_eq!(res, Ok(()));

		let res: Result<(), AppError> = try { ensure!(false, "A{value}") };
		assert_eq!(res, Err(AppError::<()>::msg("A5")));
	}

	#[test]
	fn flatten_simple() {
		let err = AppError::<()>::from_multiple([
			AppError::from_multiple([AppError::msg("A"), AppError::msg("B")]),
			AppError::msg("C"),
			AppError::from_multiple([
				AppError::from_multiple([
					AppError::msg("D"),
					AppError::from_multiple([AppError::msg("E"), AppError::msg("F")]),
				]),
				AppError::from_multiple([
					AppError::from_multiple([AppError::msg("H"), AppError::msg("I")]),
					AppError::msg("J"),
				])
				.context("G1")
				.context("G2"),
			]),
			AppError::from_multiple([AppError::msg("K")]).context("L"),
		]);

		let found = err.flatten();
		let expected = AppError::from_multiple([
			AppError::msg("A"),
			AppError::msg("B"),
			AppError::msg("C"),
			AppError::msg("D"),
			AppError::msg("E"),
			AppError::msg("F"),
			AppError::from_multiple([AppError::msg("H"), AppError::msg("I"), AppError::msg("J")])
				.context("G1")
				.context("G2"),
			AppError::msg("K").context("L"),
		]);

		assert!(
			found == expected,
			"Flattened error was wrong.\nExpected: {}\nFound: {}",
			expected.pretty(),
			found.pretty()
		);
	}
}
