use super::{Tick, linear, log};
use num_traits::Float;

/// Iterator over ticks produced by a scale.
pub struct TickIter<D> {
    inner: Box<dyn Iterator<Item = Tick<D>> + 'static>,
}

impl<D: 'static> TickIter<D> {
    pub fn new<I>(iter: I) -> Self
    where
        I: Iterator<Item = Tick<D>> + 'static,
    {
        Self {
            inner: Box::new(iter),
        }
    }

    /// Creates a `TickIter` from a vector of ticks.
    ///
    /// This is useful for providing a custom list of tick marks to a scale.
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::scale::{Tick, TickIter};
    ///
    /// let ticks = vec![
    ///     Tick { value: 0.0, level: 0 },
    ///     Tick { value: 50.0, level: 0 },
    ///     Tick { value: 100.0, level: 0 },
    /// ];
    ///
    /// let iter = TickIter::from_vec(ticks);
    /// assert_eq!(iter.count(), 3);
    /// ```
    pub fn from_vec(vec: Vec<Tick<D>>) -> Self {
        Self::new(vec.into_iter())
    }

    /// Creates an empty `TickIter` that produces no ticks.
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::scale::TickIter;
    ///
    /// let iter = TickIter::<f64>::empty();
    /// assert_eq!(iter.count(), 0);
    /// ```
    pub fn empty() -> Self {
        Self::new(std::iter::empty())
    }
}

impl<D: Float + 'static> TickIter<D> {
    pub(crate) fn from_linear(iter: linear::LinearTickIter<D>) -> Self {
        Self::new(iter)
    }

    pub(crate) fn from_log(iter: log::LogTickIter<D>) -> Self {
        Self::new(iter)
    }
}

impl<D> Iterator for TickIter<D> {
    type Item = Tick<D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
