//! Error pretty printing

// Imports
use {
	crate::{AppError, Inner},
	core::fmt,
	itertools::{Itertools, Position as ItertoolsPos},
	std::vec,
};

/// Pretty display for [`AppError`]
pub struct PrettyDisplay<'a, D = ()> {
	/// Root error
	root: &'a AppError<D>,

	/// Ignore error
	// TODO: Make this a closure?
	ignore_err: Option<fn(&AppError<D>, &D) -> bool>,
}

impl<D> fmt::Debug for PrettyDisplay<'_, D>
where
	D: fmt::Debug + 'static,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("PrettyDisplay").field("root", &self.root).finish()
	}
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum Column {
	Line,
	Empty,
}

impl Column {
	/// Returns the string for this column
	const fn as_str(self) -> &'static str {
		match self {
			Self::Line => "│ ",
			Self::Empty => "  ",
		}
	}
}

impl<'a, D> PrettyDisplay<'a, D> {
	/// Creates a new pretty display
	pub(crate) fn new(root: &'a AppError<D>) -> Self {
		Self { root, ignore_err: None }
	}

	/// Adds a callback that chooses whether to ignore an error
	pub fn with_ignore_err(self, ignore_err: fn(&AppError<D>, &D) -> bool) -> Self {
		Self {
			ignore_err: Some(ignore_err),
			..self
		}
	}

	/// Formats a single error
	fn fmt_single(&self, f: &mut fmt::Formatter<'_>, err: &AppError<D>, columns: &mut Vec<Column>) -> fmt::Result {
		// If it's multiple, display it as multiple
		let (msg, source) = match &*err.inner {
			Inner::Single { msg, source, .. } => (msg, source),
			Inner::Multiple(errs) => return self.fmt_multiple(f, errs, columns),
		};

		// Else write the top-level error
		write!(f, "{msg}")?;

		// Then, if there's a cause, write the rest
		if let Some(mut cur_source) = source.as_ref() {
			let starting_columns = columns.len();
			loop {
				// Print the pre-amble
				f.pad("\n")?;
				for c in &*columns {
					f.pad(c.as_str())?;
				}
				f.pad("└─")?;
				columns.push(Column::Empty);

				// Then check if we got to a multiple.
				match &*cur_source.inner {
					Inner::Single { msg, source, .. } => {
						write!(f, "{msg}",)?;

						// And descend
						cur_source = match source {
							Some(source) => source,
							_ => break,
						};
					},
					Inner::Multiple(errs) => {
						self.fmt_multiple(f, errs, columns)?;
						break;
					},
				}
			}
			let _: vec::Drain<'_, _> = columns.drain(starting_columns..);
		}

		Ok(())
	}

	/// Formats multiple errors
	fn fmt_multiple(&self, f: &mut fmt::Formatter<'_>, errs: &[AppError<D>], columns: &mut Vec<Column>) -> fmt::Result {
		// Write the top-level error
		write!(f, "Multiple errors:")?;

		// For each error, write it
		let mut ignored_errs = 0;
		for (pos, err) in errs.iter().with_position() {
			// If we should ignore the error, skip
			if let Some(ignore_err) = self.ignore_err &&
				self::should_ignore(err, ignore_err)
			{
				ignored_errs += 1;
				continue;
			}

			f.pad("\n")?;
			for c in &*columns {
				f.pad(c.as_str())?;
			}

			// Note: We'll only print `└─` if we have no ignored errors, since if we do,
			//       we need that to print the final line showcasing how many we ignored
			match ignored_errs == 0 && matches!(pos, ItertoolsPos::Last | ItertoolsPos::Only) {
				true => {
					f.pad("└─")?;
					columns.push(Column::Empty);
				},
				false => {
					f.pad("├─")?;
					columns.push(Column::Line);
				},
			}

			self.fmt_single(f, err, columns)?;
			let _: Option<_> = columns.pop();
		}

		if ignored_errs != 0 {
			f.pad("\n")?;
			for c in &*columns {
				f.pad(c.as_str())?;
			}
			f.pad("└─")?;
			write!(f, "({ignored_errs} ignored errors)")?;
		}

		Ok(())
	}
}

impl<D> fmt::Display for PrettyDisplay<'_, D> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let mut columns = vec![];
		self.fmt_single(f, self.root, &mut columns)?;
		assert_eq!(columns.len(), 0, "There should be no columns after formatting");

		Ok(())
	}
}

// Returns whether an error should be ignored
fn should_ignore<D>(err: &AppError<D>, ignore_err: fn(&AppError<D>, &D) -> bool) -> bool {
	match &*err.inner {
		// When dealing with a single error, we ignore if it any error in it's tree, including itself
		// should be ignored.
		Inner::Single { source, data, .. } =>
			ignore_err(err, data) ||
				source
					.as_ref()
					.is_some_and(|source| self::should_ignore(source, ignore_err)),

		// For multiple errors, we only ignore it if all should be ignored.
		Inner::Multiple(errs) => errs.iter().all(|err| self::should_ignore(err, ignore_err)),
	}
}
