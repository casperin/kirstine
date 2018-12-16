/// # Variance
///
/// Implementation of [two pass
/// algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm)
/// for calculating the variance of a sample.
///
/// **Notice** there is a difference between `sample::variance` and `population::variance`. This
/// function uses `N - 1` as divisor because it deals with a sample.
///
/// ## Example
/// ```
/// let sample = vec![600.0, 470.0, 170.0, 430.0, 300.0];
/// let mu = kirstine::mean(&sample);
/// assert_eq!(kirstine::sample::variance(&sample, mu), 27130.0);
/// ```
pub fn variance(sample: &Vec<f64>, mu: f64) -> f64 {
    let tss = super::sum_of_squares(&sample, mu);
    tss / (sample.len() - 1) as f64
}

/// # Standard deviation
///
/// Difined as the square root of the deviation.
///
/// Again notice that that is a difference between `sample::standard_deviation` and
/// `population::standard_deviation`, as they make use of different `variance` functions.
pub fn standard_deviation(data: &Vec<f64>, mu: f64) -> f64 {
    variance(&data, mu).sqrt()
}

pub fn z_score(mu_sample: f64, n_sample: usize, mu_population: f64, sigma_population: f64) -> f64 {
    let dividend = mu_sample - mu_population;
    let divisor = sigma_population / (n_sample as f64).sqrt();
    dividend / divisor
}

pub fn z_score_single_sample(sample: f64, mu: f64, sigma: f64) -> f64 {
    z_score(sample, 1, mu, sigma)
}

#[test]
fn z_score_single_sample_test() {
    let sample1 = 105.0;
    let mu1 = 100.0;
    let sigma1 = 4.0;
    let result1 = z_score_single_sample(sample1, mu1, sigma1);
    assert_eq!(result1, 1.25);

    let sample2 = 100.0;
    let mu2 = 100.0;
    let sigma2 = 4.0;
    let result2 = z_score_single_sample(sample2, mu2, sigma2);
    assert_eq!(result2, 0.0);
}
