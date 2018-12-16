pub mod population;
pub mod sample;

use std::cmp::Ordering::Less;
use std::collections::HashMap;

/// # Arithmetic Mean
///
/// Calculates the mean, or the average, of a vector of floats.
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Mean)
///
/// Panics if dataset is empty.
///
/// ## Example
/// ```
/// let data = vec![1.0, 3.0, 3.0, 2.0, 1.0];
/// assert_eq!(kirstine::mean(&data), 2.0);
/// ```
pub fn mean(data: &Vec<f64>) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

/// # Arithmetic Median
///
/// Calculates the median.
///
/// The function needs to clone and sort the dataset which is expensive, so if you know that your
/// dataset is sorted, then use `kirstine::median_from_sorted` instead.
///
/// Panics if dataset is empty.
///
/// ## Example
/// ```
/// let data = vec![2.0, 5.0, 1.0];
/// assert_eq!(kirstine::median(&data), 2.0);
///
/// let data = vec![2.0, 5.0, 3.0, 1.0];
/// assert_eq!(kirstine::median(&data), 2.5);
/// ```
pub fn median(data: &Vec<f64>) -> f64 {
    let mut copy = data.clone();
    copy.sort_by(|m, n| m.partial_cmp(n).unwrap_or(Less));
    median_from_sorted(&copy)
}

/// # Median from sorted vector
///
/// Faster version of `kirstine::median` that can be used if you know that your dataset is already
/// sorted.
///
/// Panics if dataset is empty.
pub fn median_from_sorted(data: &Vec<f64>) -> f64 {
    if data.is_empty() {
        panic!("Can not find median of empty list");
    }
    if data.len() % 2 == 1 {
        data[((data.len() - 1) / 2) as usize]
    } else {
        let upper_index = data.len() / 2 as usize;
        let lower_index = upper_index - 1;
        let sum = data[upper_index] + data[lower_index];
        sum / 2.0
    }
}

/// # Mode
///
/// Finds the *mode* of a dataset.
///
/// Panics if dataset is empty.
///
/// ## Example
/// ```
/// let data = vec![2.0, 5.0, 1.0, 3.0, 1.0];
/// assert_eq!(kirstine::mode(&data), &1.0);
/// ```
pub fn mode(data: &Vec<f64>) -> &f64 {
    if data.is_empty() {
        panic!("Can not find mode of empty list");
    }
    let mut nums = HashMap::new();
    for n in data.iter() {
        let n_str = n.to_string();
        let (count, _) = nums.entry(n_str).or_insert((0, n));
        *count += 1;
    }
    nums.values().max_by(|(a, _), (b, _)| a.cmp(b)).unwrap().1
}

/// # Arithmetic Range
///
/// Calculates the arithmetic range, and coefficient of range, of a dataset.
///
/// The return value is a tuple of the range, the coefficient of range, and a tuple of largest and
/// smallest value: `(range, coef_of_range, (&smallest, &largest))`
///
/// ## Example
/// ```
/// let data = vec![89.0, 73.0, 84.0, 91.0, 87.0, 77.0, 94.0];
/// let (range, coef_of_range, (smallest, largest)) = kirstine::range(&data);
/// assert_eq!(range, 21.0);
/// assert!((coef_of_range - 0.1257).abs() < 0.0001);
/// assert_eq!(smallest, &73.0);
/// assert_eq!(largest, &94.0);
/// ```
pub fn range(data: &Vec<f64>) -> (f64, f64, (&f64, &f64)) {
    if data.is_empty() {
        panic!("Can not find range of empty list");
    }
    let mut largest = &data[0];
    let mut smallest = &data[0];
    for x in data.iter() {
        if x > largest {
            largest = x;
        }
        if x < smallest {
            smallest = x;
        }
    }
    let range = largest - smallest;
    let coef_of_range = range / (largest + smallest);
    (range, coef_of_range, (smallest, largest))
}

/**
 * # Pearson correlation coefficient
 *
 * This is what is normally referred to when talking about finding the correlation.
 *
 * The `data` is a vector of tuples with `x` and `y` data points. Returns a number between -1
 * and 1.
 *
 * [Wikipedia article](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
 */
pub fn correlation(data: &Vec<(f64, f64)>) -> f64 {
    let n = data.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    for (x, y) in data.iter() {
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x.powf(2.0);
        sum_y2 += y.powf(2.0);
    }
    let dividend = n * sum_xy - sum_x * sum_y;
    let divisor_left = n * sum_x2 - sum_x.powf(2.0);
    let divisor_right = n * sum_y2 - sum_y.powf(2.0);
    let divisor = (divisor_left * divisor_right).sqrt();
    dividend / divisor
}

#[test]
fn correlation_test() {
    let data = vec![
        (43.0, 99.0),
        (21.0, 65.0),
        (25.0, 79.0),
        (42.0, 75.0),
        (57.0, 87.0),
        (59.0, 81.0),
    ];
    let result = correlation(&data);
    let expected = 0.529809;
    let diff = (expected - result).abs();
    assert!(diff < 0.0000001);
}

/// # Sum of squares
///
/// The sum, over all observations, of the squared differences of each observation from the overall
/// mean.
pub fn sum_of_squares(data: &Vec<f64>, mu: f64) -> f64 {
    data.iter().map(|n| (n - mu).powf(2.0)).sum()
}

/// # Chi-squared test
///
/// Takes a vector of tuples with expected as first item, and observed as second.
///
/// [Wikipedia article](https://en.wikipedia.org/wiki/Chi-squared_test).
///
/// ## Example
/// ```
/// let data = vec![
///     (25.0, 23.0),
///     (16.0, 20.0),
///     (4.0, 3.0),
///     (24.0, 24.0),
///     (8.0, 10.0),
/// ];
/// assert_eq!(kirstine::chi_squared(&data), 1.91);
/// ```
pub fn chi_squared(data: &Vec<(f64, f64)>) -> f64 {
    data.iter().map(|(e, o)| (e - o).powf(2.0) / e).sum()
}

#[test]
fn chi_squared_test() {
    let data = vec![
        (21.33333334, 29.0),
        (21.33333334, 24.0),
        (21.33333334, 22.0),
        (21.33333334, 19.0),
        (21.33333334, 21.0),
        (21.33333334, 18.0),
        (21.33333334, 19.0),
        (21.33333334, 20.0),
        (21.33333334, 23.0),
        (21.33333334, 18.0),
        (21.33333334, 20.0),
        (21.33333334, 23.0),
    ];
    let result = chi_squared(&data);
    let expected = 5.09375;
    let diff = (result - expected).abs();
    assert!(diff < 0.0000001);
}
