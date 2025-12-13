# App error

This crate provides an error type, `AppError`, that is intended for usage in applications.

It is `Send`, `Sync`, `'static`, and, importantly, cheaply `Clone`-able.

To achieve this, it serializes every error it receives without owning it, meaning that
you also can't retrieve the error later by downcasting it.

It is also able to store multiple errors at once and provide pretty-printing of all
of these them.

It can carry an optional data parameter that may be retrieved later on.

## Examples

````rust
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
````
