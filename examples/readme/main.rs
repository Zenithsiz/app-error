use app_error::{AppError, Context, app_error};

fn fallible_fn1() -> Result<(), AppError> {
	// Create an error
	Err(app_error!("Fn1 failed!"))
}

fn fallible_fn2() -> Result<(), AppError> {
	// Add context to results
	fallible_fn1().context("Fn2 failed!")
}

fn main() {
	let err = fallible_fn2().expect_err("Will return an error");

	// Pretty printing:
	// ```
	// Fn2 failed!
	// └─Fn1 failed!
	// ```
	println!("{err:?}");
}
