use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;
use csv;
use plotters::prelude::*;
use serde::Deserialize;

// Struct to represent a single data point read from the spline data CSV.
#[derive(Debug, Deserialize)]
struct DataPoint {
    curve: u32,
    index: u32,
    position: f64,
    value: f64,
    derivative: Option<f64>,
}

/// Struct to hold spline data points and endpoint derivatives.
struct SplineData {
    positions: Vec<f64>,
    values: Vec<f64>,
    derivative_start: Option<f64>, // If Some, use clamped condition at start
    derivative_end: Option<f64>, // If Some, use clamped condition at end
}

/// Computes the second derivatives for the spline interpolation.
fn compute_second_derivative(
    positions: &[f64],
    values: &[f64],
    derivative_start: Option<f64>,
    derivative_end: Option<f64>,
) -> Vec<f64> {
    // Calculate number of intervals between nodes.
    let intervals = positions.len() - 1;
    // Compute the width of each interval.
    let mut interval_widths = Vec::with_capacity(intervals);
    for i in 0..intervals {
        interval_widths.push(positions[i + 1] - positions[i]);
    }

    // Total number of nodes.
    let nodes = intervals + 1;
    // Coefficients for the tridiagonal system.
    let mut a = vec![0.0; nodes];
    let mut b = vec![0.0; nodes];
    let mut c = vec![0.0; nodes];
    let mut d = vec![0.0; nodes];

    // Setup the equations for the left endpoint.
    if let Some(derivative_start) = derivative_start {
        // Clamped condition at the left endpoint.
        b[0] = 2.0 * interval_widths[0];
        c[0] = interval_widths[0];
        // d[0] contains the difference between the slope of the first segment
        // and the given derivative.
        d[0] = 6.0 * ((values[1] - values[0])
                      / interval_widths[0] - derivative_start);
    } else {
        // Natural spline: Second derivative at left endpoint equals 0.
        b[0] = 1.0;
        d[0] = 0.0;
    }

    // Setup the equations for the interior points.
    for i in 1..intervals {
        a[i] = interval_widths[i - 1];
        b[i] = 2.0 * (interval_widths[i - 1] + interval_widths[i]);
        c[i] = interval_widths[i];
        d[i] = 6.0 * ((values[i + 1] - values[i]) / interval_widths[i]
                      - (values[i] - values[i - 1]) / interval_widths[i - 1]);
    }

    // Setup the equations for the right endpoint.
    if let Some(derivative_end) = derivative_end {
        // Clamped condition at the right endpoint.
        a[intervals] = interval_widths[intervals - 1];
        b[intervals] = 2.0 * interval_widths[intervals - 1];
        d[intervals] = 6.0 * (derivative_end
                                  - (values[intervals] - values[intervals - 1])
                                  / interval_widths[intervals - 1]);
    } else {
        // Natural spline: Second derivative at right endpoint equals 0.
        b[intervals] = 1.0;
        d[intervals] = 0.0;
    }

    // Solve the tridiagonal system using the Thomas algorithm.
    // Forward elimination: modify the coefficients.
    for i in 1..nodes {
        let factor = a[i] / b[i - 1];
        b[i] -= factor * c[i - 1];
        d[i] -= factor * d[i - 1];
    }

    // Back substitution: compute the second derivatives.
    let mut second_derivatives = vec![0.0; nodes];
    second_derivatives[nodes - 1] = d[nodes - 1] / b[nodes - 1];
    for i in (0..nodes - 1).rev() {
        second_derivatives[i] = (d[i] - c[i] * second_derivatives[i + 1]) / b[i];
    }

    second_derivatives
}

/// Evaluate the cubic spline at a given evaluation point using the computed second derivatives.
fn evaluate_spline(
    positions: &[f64],
    values: &[f64],
    second_derivatives: &[f64],
    eval_point: f64,
) -> f64 {
    let nodes = positions.len();
    // Determine which interval contains the evaluation point.
    let mut segment_index = 0;
    while segment_index < nodes - 1 && eval_point > positions[segment_index + 1] {
        segment_index += 1;
    }

    // Ensure that we stay within bounds.
    if segment_index >= nodes - 1 {
        segment_index = nodes - 2;
    }

    // Calculate interval width for the selected segments.
    let interval_width = positions[segment_index + 1] - positions[segment_index];
    // Compute distance from the left node of the interval.
    let dist_into_interval = eval_point - positions[segment_index];
    // Compute weights for interpolation.
    let weight_a = (positions[segment_index + 1] - eval_point) / interval_width;
    let weight_b = dist_into_interval / interval_width;
    // Compute the spline value using the standard cubic spline formula.
    let spline_value = weight_a * values[segment_index]
        + weight_b * values[segment_index + 1]
        + ((weight_a.powi(3) - weight_a) * second_derivatives[segment_index]
           + (weight_b.powi(3) - weight_b) * second_derivatives[segment_index + 1])
        * (interval_width * interval_width) / 6.0;

    spline_value
}

/// Samples the spline over each interval with the given sampling step,
/// returning a vector of (x, S(x)) points.
fn sample_spline(
    positions: &[f64],
    values: &[f64],
    second_derivatives: &[f64],
    sampling_step: f64,
) -> Vec<(f64, f64)> {
    let mut sampled_points = Vec::new();
    let start_position = positions[0];
    let end_position = *positions.last().unwrap();
    let mut sample_position = start_position;

    // Sample the spline from the start to the end position.
    while sample_position <= end_position {
        sampled_points.push((sample_position, evaluate_spline(
            positions, values, second_derivatives, sample_position
        )));
        sample_position += sampling_step;
    }

    // Ensure the final point is included.
    if sampled_points.last().map_or(true, |&(pos, _)| (pos - end_position).abs() > 1e-6) {
        sampled_points.push((end_position, evaluate_spline(
            positions, values, second_derivatives, end_position
        )));
    }

    sampled_points
}

/// Plots the given spline curves and their original data points into an image.
fn plot_spline(
    filename: &str,
    sampled_points: &[Vec<(f64, f64)>],
    original_points: &[&[(f64, f64)]],
    caption: &str,
) -> Result<(), Box<dyn Error>> {
    // Check for the "images" directory and create it if necessary.
    let images_directory = "images";
    if !Path::new(images_directory).exists() {
        fs::create_dir_all(images_directory)?;
    }
    // Build the full file path for the image.
    let filepath = format!("{}/{}", images_directory, filename);

    // Setup the drawing area with defined x and y ranges.
    let x_range = 1.0..30.0;
    let y_range = 2.5..7.5;
    let drawing_area = BitMapBackend::new(&filepath, (1280, 720)).into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&drawing_area)
        .margin(20)
        .caption(caption, ("sans-serif", 30))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;
    chart.configure_mesh().draw()?;

    // Define colors for each curve.
    let colors = [&RED, &GREEN, &BLUE];
    // Draw the sampled spline curves.
    for (i, spline_curve) in sampled_points.iter().enumerate() {
        chart.draw_series(LineSeries::new(
            spline_curve.clone(),
            colors[i % colors.len()].stroke_width(2),
        ))?;
    }

    // Draw the original data points on top of the spline curves.
    for (i, points) in original_points.iter().enumerate() {
        chart.draw_series(points.iter().map(|&(x_val, y_val)| {
            Circle::new((x_val, y_val), 4, colors[i % colors.len()].filled())
        }))?;
    }
    drawing_area.present()?;
    println!("Plot saved to {}", filepath);

    Ok(())
}

/// Read a CSV file, groups the data by curve, and returns a vector of SplineData structs.
fn read_csv() -> Result<Vec<SplineData>, Box<dyn Error>> {
    let datapath = "data/spline_data.csv";

    // Ensure the data directory exists.
    if !Path::new("data").exists() {
        fs::create_dir_all("data")?;
    }

    // Open and parse the CSV file.
    let mut reader = csv::Reader::from_path(datapath)?;
    let mut curves_map: HashMap<u32, Vec<DataPoint>> = HashMap::new();
    for result in reader.deserialize() {
        let record: DataPoint = result?;
        curves_map.entry(record.curve).or_default().push(record);
    }

    let mut data_vector = Vec::new();
    // Process each curve and sort by index.
    for (_curve_id, mut points) in curves_map {
        points.sort_by_key(|p| p.index);
        let positions = points.iter().map(|p| p.position).collect();
        let values = points.iter().map(|p| p.value).collect();
        let derivative_start = points.first().and_then(|p| p.derivative);
        let derivative_end = points.last().and_then(|p| p.derivative);
        data_vector.push(SplineData {
            positions,
            values,
            derivative_start,
            derivative_end,
        });
    }

    Ok(data_vector)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Read spline data from the CSV file.
    let data_vector = read_csv()?;

    // Collect original points from SplineData for plotting.
    let mut original_points: Vec<Vec<(f64, f64)>> = Vec::new();
    for data in &data_vector {
        let points: Vec<(f64, f64)> = data.positions
            .iter()
            .zip(data.values.iter())
            .map(|(&pos, &val)| (pos, val))
            .collect();
        original_points.push(points);
    }

    // Convert to slice-of-slices required by the plot_spline function.
    let original_points_refs: Vec<&[(f64, f64)]> = original_points
        .iter()
        .map(|v| v.as_slice())
        .collect();

    // Clamped Splines (Exercise 27)
    // Compute clamped second derivatives and sample the spline curves.
    let mut clamped_curves = Vec::new();
    for data in &data_vector {
        let second_derivatives = compute_second_derivative(
            &data.positions,
            &data.values,
            data.derivative_start,
            data.derivative_end,
        );
        let sampled = sample_spline(
            &data.positions,
            &data.values,
            &second_derivatives,
            0.1,
        );
        clamped_curves.push(sampled);
    }

    // Plot clamped cubic spline curves.
    plot_spline(
        "clamped.png",
        &clamped_curves,
        &original_points_refs,
        "Clamped Cubic Spline",
    )?;

    // Natural Splines (Exercise 28)
    // Compute natural second derivatives and sample the spline curves.
    let mut natural_curves = Vec::new();
    for data in &data_vector {
        let second_derivatives = compute_second_derivative(
            &data.positions,
            &data.values,
            None,
            None,
        );
        let sampled = sample_spline(
            &data.positions,
            &data.values,
            &second_derivatives,
            0.1,
        );
        natural_curves.push(sampled);
    }

    plot_spline(
        "natural.png",
        &natural_curves,
        &original_points_refs,
        "Natural Cubic Spline",
    )?;

    Ok(())
}
