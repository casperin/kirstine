// https://www.stattrek.com/statistics/formulas.aspx
// https://www.dummies.com/education/math/statistics/top-10-statistical-formulas/

/// # Variance
///
/// Implementation of [two pass
/// algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm)
/// for calculating the variance of a full population.
///
/// **Notice** there is a difference between `population::variance` and `sample::variance`. This
/// function uses `N` as divisor since it deals with the whole population, and not just a sample.
///
/// ## Example
/// ```
/// let population = vec![600.0, 470.0, 170.0, 430.0, 300.0];
/// let mu = kirstine::mean(&population);
/// assert_eq!(kirstine::population::variance(&population, mu), 21704.0);
/// ```
pub fn variance(population: &Vec<f64>, mu: f64) -> f64 {
    let tss = super::sum_of_squares(&population, mu);
    tss / population.len() as f64
}

/// # Standard deviation
///
/// Difined as the square root of the deviation.
///
/// Again notice that that is a difference between `population::standard_deviation` and
/// `sample::standard_deviation`, as they make use of different `variance` functions.
pub fn standard_deviation(dataset: &Vec<f64>, mu: f64) -> f64 {
    variance(&dataset, mu).sqrt()
}
