//! Multiple error handling

// Imports
use {
	crate::AppError,
	itertools::Itertools,
	std::{
		fmt,
		ops::{ControlFlow, FromResidual, Try},
	},
};

/// Helper type to collect a `IntoIter<Item = Result<T, AppError>>`
/// into a `Result<C, AppError>` with all of the errors instead of the first.
pub enum AllErrs<C, D = ()> {
	Ok(C),
	Err(Vec<AppError<D>>),
}

impl<C, D> fmt::Debug for AllErrs<C, D>
where
	C: fmt::Debug,
	D: fmt::Debug + 'static,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Ok(value) => f.debug_tuple("Ok").field(value).finish(),
			Self::Err(errs) => f.debug_tuple("Err").field(errs).finish(),
		}
	}
}

impl<C, D> Default for AllErrs<C, D>
where
	C: Default,
{
	fn default() -> Self {
		Self::Ok(C::default())
	}
}
impl<C, U, D> Extend<Result<U, AppError<D>>> for AllErrs<C, D>
where
	C: Extend<U>,
{
	fn extend<T: IntoIterator<Item = Result<U, AppError<D>>>>(&mut self, iter: T) {
		// TODO: Do this more efficiently?
		for res in iter {
			match (&mut *self, res) {
				// If we have a collection, and we get an item, extend it
				(Self::Ok(collection), Ok(item)) => collection.extend_one(item),
				// If we have a collection, but find an error, switch to errors
				(Self::Ok(_), Err(err)) => *self = Self::Err(vec![err]),
				// If we have errors and got an item, ignore it
				(Self::Err(_), Ok(_)) => (),
				// If we have errors and got an error, extend it
				(Self::Err(errs), Err(err)) => errs.push(err),
			}
		}
	}
}

impl<C, T, D> FromIterator<Result<T, AppError<D>>> for AllErrs<C, D>
where
	C: Default + Extend<T>,
{
	fn from_iter<I>(iter: I) -> Self
	where
		I: IntoIterator<Item = Result<T, AppError<D>>>,
	{
		// TODO: If we get any errors, don't allocate memory for the rest of the values?
		let (values, errs) = iter.into_iter().partition_result::<C, Vec<_>, _, _>();
		match errs.is_empty() {
			true => Self::Ok(values),
			false => Self::Err(errs),
		}
	}
}

pub struct AllErrsResidue<D>(Vec<AppError<D>>);

impl<D> fmt::Debug for AllErrsResidue<D>
where
	D: fmt::Debug + 'static,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_tuple("AllErrsResidue").field(&self.0).finish()
	}
}

impl<C, D> Try for AllErrs<C, D> {
	type Output = C;
	type Residual = AllErrsResidue<D>;

	fn from_output(output: Self::Output) -> Self {
		Self::Ok(output)
	}

	fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
		match self {
			Self::Ok(values) => ControlFlow::Continue(values),
			Self::Err(errs) => ControlFlow::Break(AllErrsResidue(errs)),
		}
	}
}

impl<T, D> FromResidual<AllErrsResidue<D>> for AllErrs<T, D> {
	fn from_residual(residual: AllErrsResidue<D>) -> Self {
		Self::Err(residual.0)
	}
}

impl<T, D> FromResidual<AllErrsResidue<D>> for Result<T, AppError<D>> {
	fn from_residual(residual: AllErrsResidue<D>) -> Self {
		let err = match <[_; 1]>::try_from(residual.0) {
			Ok([err]) => err,
			Err(errs) => {
				assert!(!errs.is_empty(), "`ResultMultipleResidue` should hold at least 1 error");
				AppError::from_multiple(errs)
			},
		};

		Err(err)
	}
}
