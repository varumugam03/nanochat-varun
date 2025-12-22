use std::collections::HashMap as StdHashMap;
use fancy_regex::Regex;
use pyo3::prelude::*;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;


// GPT-4 style regex pattern for tokenization
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

type Pair = (u32, u32);

#[pyclass]
pub struct Tokenizer {
    /// Maps pairs of tokens to their merged token ID
    pub merges : StdHashMap<Pair, u32>,
    /// The regex pattern used for tokenization
    pub pattern: String,
    /// The compiled regex for efficiency
    compiled_pattern: Regex,
}

// ------internal helpers------

#[derive(Clone, Debug)] // tell compiler to autowrite these methods
struct Word {
    ids : Vec<u32>,
}

impl Word {
    #[inline] // hint to compiler that this function is small and called often enough to just copy code "in line"
    fn new(ids:Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(pair, u32)> {
        let (a, b) = pair;

    }
}

#[inline]
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words.par_iter().enumerate()
            .fold(
                || (AHashMap::new(), AHashMap::new()),
                |(mut acc_pc, mut acc_wtu), (i, w)| {
                    if w.ids.len() >= 2 && counts[i] != 0 {
                        for (a, b) in w.pairs() {
                            *acc_pc.entry((a, b)).or_default() += counts[i];
                            acc_wtu.entry((a, b)).or_default().insert(i);
                        }
                    }
                    (acc_pc, acc_wtu)
                },
            )
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut global_pc, mut global_wtu), (local_pc, local_wtu)| {
                    for (k, v) in local_pc {
                        *global_pc.entry(k).or_default() += v;
                    }
                    for (k, s) in local_wtu {
                        global_wtu.entry(k).or_default().extend(s);
                    }
                    (global_pc, global_wtu)
                },
            )
}

// ------end helpers------

// ------rust only tokenizer impl------

impl Tokenizer {

    /// Core incremental BPE training - given unique words and their counts.
    /// `words` : one entry per unique chunk of (Vec<u32> of token-ids/bytes).
    /// `counts` : same length as `words`, count per chunk

    fn train_core_incremental(&mut self, mut words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
        assert!(vocab_size >= 256, "vocab_size must be at least 256");
        let num_merges = vocab_size - 256;
        log::info!("Training BPE with {} merges", num_merges);
        self.merges.clear();

        // initial pair_counts and where_to_update (parallel)
        log::info!("Computing initial pair counts form {} unique sequences", words.len());
        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);
    }

}

// ------python interface------
#[pymethods]
impl Tokenizer {
    ///Create a new Tokenizer
    #[new]
    pub fn new() -> Self {
        Self {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").expect("Empty regex pattern should be valid"),
        }
    }

    ///Train from a streaming iterator (parallel ingestion)
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None))]
    #[pyo3(text_signature = "(self, iterator, vocab_size, buffer_size=8192, pattern=None)")]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
    ) -> PyResult<StdHashMap<String, i32>> {// TODO : CHANGE THIS BACK TO NOTHING -- ONLY IN HERE TO EXPOSE TO PYTHON AND TEST INITIALLY
        //Use provided pattern or default to the GPT-4 pattern
        let pattern_str = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());

        self.pattern = pattern_str.clone();
        //mapping the error to a python exception and the ? is just short hand for exiting the function early if there is a error and returning
        self.compiled_pattern = Regex::new(&pattern_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid regex pattern: {}", e)))?;

        //prepare a true python iterator object - gemini inferred this is probably so this variable can outlive the GIL lifetime
        //the other option would have been to do iterator.iter() in pyo3 but that would have been tied to the GIL lifetime
        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        //global chunk counts
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();
        
        // temporary buffer we refill under the GIL - the 8192 thing
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!("Processing sequences from iterator (buffer_size: {})", buffer_size);
        let mut total_sequences = 0u64;

        // Helper: refill `buf` with up to `buffer_size` strings from the Python iterator
        // Returns Ok(True) if the iterator is exhausted, Ok(false) otherwise.
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::with_gil(|py| { 
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py));
                            }else{
                                return Ok(true)
                            }
                        }
                    }
                    
                }
            })
        };

        //Stream ingestion loop: refill under GIL, process without GIL (parallel)
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_sequences += buf.len() as u64;

            let pattern = self.compiled_pattern.clone();
            let local: AHashMap<CompactString, i32> = py.allow_threads(|| { // py.allow_threads releases the GIL
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                        for mat in pattern.find_iter(s) {
                            let piece = mat.expect("Regex match failed").as_str();
                            *m.entry(CompactString::from(piece)).or_default() += 1;
                        }
                        m
                    })
                    .reduce(
                        || AHashMap::new(),
                        |mut a, b| {
                            for (k,v) in b {
                                *a.entry(k).or_default() += v;
                            }
                            a
                        },
                    )
            });

            //merge into the global map
            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }

            if exhausted {
                break;
            }
        }
        log::info!("Processed {} sequences total, {} unique", total_sequences, counts.len());

        //UNCOMMENT THIS LATER
        // //Materialize words & counts
        // let mut words = Vec::with_capacity(counts.len());
        // let mut cvec = Vec::with_capacity(counts.len());
        // for (chunk, c) in counts.into_iter() {
        //     words.push(Word::new(chunk.as_bytes().iter().map(|&b| b as u32).collect()));
        //     cvec.push(c);
        // }

        //CONVERT AHashMap --> StdHashMap for testing -- REMOVE AFTER
        let result_map: StdHashMap<String, i32> = counts.into_iter().map(|(k,v)| (k.to_string(), v)).collect();
        
        Ok(result_map)
    }
}

#[pymodule]
fn _rustbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); //forwards Rust `log` to Python's `logging`
    m.add_class::<Tokenizer>()?;
    Ok(())
}
