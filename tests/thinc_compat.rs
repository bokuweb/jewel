use jewel_core::{affine_softmax, expand_window, hash_embed, layer_norm, maxout, Matrix};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    hash_embed: HashEmbedCase,
    maxout: MaxoutCase,
    layer_norm: LayerNormCase,
    expand_window: WindowCase,
    softmax: SoftmaxCase,
}

#[derive(Deserialize)]
struct HashEmbedCase {
    ids: Vec<u64>,
    seed: u32,
    embeddings: Vec<Vec<f32>>,
    expected: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct MaxoutCase {
    input: Vec<Vec<f32>>,
    weights: Vec<Vec<Vec<f32>>>,
    bias: Vec<Vec<f32>>,
    expected: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct LayerNormCase {
    input: Vec<Vec<f32>>,
    gain: Vec<f32>,
    bias: Vec<f32>,
    expected: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct WindowCase {
    input: Vec<Vec<f32>>,
    window: usize,
    lengths: Vec<usize>,
    expected: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct SoftmaxCase {
    input: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    expected: Vec<Vec<f32>>,
}

fn matrix(rows: Vec<Vec<f32>>) -> Matrix {
    let row_count = rows.len();
    let column_count = rows.first().map_or(0, Vec::len);
    assert!(rows.iter().all(|row| row.len() == column_count));
    Matrix::new(
        row_count,
        column_count,
        rows.into_iter().flatten().collect(),
    )
    .unwrap()
}

fn assert_close(actual: &Matrix, expected: Vec<Vec<f32>>) {
    let expected = matrix(expected);
    assert_eq!(
        (actual.rows(), actual.cols()),
        (expected.rows(), expected.cols())
    );
    for (index, (actual, expected)) in actual
        .as_slice()
        .iter()
        .zip(expected.as_slice())
        .enumerate()
    {
        let tolerance = 1e-6_f32.max(expected.abs() * 1e-6);
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/thinc_ops_3_8.json")).unwrap()
}

#[test]
fn hash_embed_matches_thinc_3_8() {
    let case = fixture().hash_embed;
    let output = hash_embed(&case.ids, case.seed, &matrix(case.embeddings)).unwrap();
    assert_close(&output, case.expected);
}

#[test]
fn maxout_matches_thinc_3_8() {
    let case = fixture().maxout;
    let outputs = case.weights.len();
    let pieces = case.weights.first().map_or(0, Vec::len);
    let weights = case
        .weights
        .into_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();
    let bias = case.bias.into_iter().flatten().collect::<Vec<_>>();
    let output = maxout(&matrix(case.input), &weights, &bias, outputs, pieces).unwrap();
    assert_close(&output, case.expected);
}

#[test]
fn layer_norm_matches_thinc_3_8() {
    let case = fixture().layer_norm;
    let output = layer_norm(&matrix(case.input), &case.gain, &case.bias).unwrap();
    assert_close(&output, case.expected);
}

#[test]
fn expand_window_matches_thinc_3_8() {
    let case = fixture().expand_window;
    let output = expand_window(&matrix(case.input), case.window, &case.lengths).unwrap();
    assert_close(&output, case.expected);
}

#[test]
fn softmax_matches_thinc_3_8() {
    let case = fixture().softmax;
    let outputs = case.weights.len();
    let weights = case.weights.into_iter().flatten().collect::<Vec<_>>();
    let output = affine_softmax(
        &matrix(case.input),
        &weights,
        &case.bias,
        outputs,
        true,
        1.0,
    )
    .unwrap();
    assert_close(&output, case.expected);
}
