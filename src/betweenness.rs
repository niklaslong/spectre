//! A module for performing the multi-threaded computation of betweenness

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use crate::graph::GraphIndex;

const MIN_NUM_THREADS: usize = 1;
const MAX_NUM_THREADS: usize = 128;

/// this is an implementation of Ulrik Brandes's
/// A Faster Algorithm for Betweenness Centrality
/// http://snap.stanford.edu/class/cs224w-readings/brandes01centrality.pdf
/// page 10, "Algorithm 1: Betweenness centrality in unweighted graphs"
fn betweenness_for_node(
    index: usize,
    indices: &Vec<Vec<GraphIndex>>,
    betweenness_count: &mut [f64],
) {
    let num_nodes = indices.len();

    let mut sigma: Vec<f64> = vec![0.0; num_nodes];
    let mut distance: Vec<usize> = vec![num_nodes + 1; num_nodes];
    let mut totals: Vec<Vec<usize>> = vec![Vec::<usize>::new(); num_nodes];
    let mut delta: Vec<f64> = vec![0.0; num_nodes];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut stack: Vec<usize> = Vec::new();

    sigma[index] = 1.0;
    distance[index] = 0;
    queue.push_back(index);

    while !queue.is_empty() {
        let v = queue.pop_front().unwrap();
        stack.push(v);

        for w in &indices[v] {
            if distance[*w as usize] == num_nodes + 1 {
                distance[*w as usize] = distance[v] + 1;
                queue.push_back(*w as usize);
            }
            if distance[*w as usize] == distance[v] + 1 {
                sigma[*w as usize] += sigma[v];
                totals[*w as usize].push(v);
            }
        }
    }

    while !stack.is_empty() {
        let w = stack[stack.len() - 1];
        stack.pop();

        for j in 0..totals[w].len() {
            let v = totals[w][j];
            delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w]);
        }
        if w != index {
            betweenness_count[w] += delta[w];
        }
    }
}

/// this function is the thread task
/// grabs next unprocessed node
/// if no more nodes, exits
/// returning betweenness
fn betweenness_task(acounter: Arc<Mutex<usize>>, aindices: Arc<Vec<Vec<GraphIndex>>>) -> Vec<f64> {
    let start = Instant::now();
    let indices = &aindices;
    let num_nodes = indices.len();

    // each worker thread keeps its own cache of data
    // these are returned when the thread finishes
    // and then summed by the caller
    let mut betweenness_count: Vec<f64> = vec![0.0; num_nodes];

    let mut finished = false;
    while !finished {
        let mut counter = acounter.lock().unwrap();
        let index: usize = *counter;
        *counter += 1;
        drop(counter);
        if index < num_nodes {
            if index % 100 == 0 {
                println!("node: {}, time: {:?}", index, start.elapsed());
            }
            betweenness_for_node(index, indices, &mut betweenness_count);
        } else {
            finished = true;
        }
    }
    betweenness_count
}

/// This public function is called by the graph method
/// closeness_centrality.  It does all
/// the heavy lifting with processing the data via
/// multiple threads
/// It is reponsibility for:
/// - setting up the data to be passed to the threads
/// - instantiating and spawning the threads
/// - collecting the results when each is finished
/// - added the results together, and returning them
pub fn compute_betweenness(
    indices: Vec<Vec<GraphIndex>>,
    mut num_threads: usize,
    normalize: bool,
) -> Vec<f64> {
    println!("compute betweennes normalize {normalize}");
    let start = Instant::now();
    num_threads = num_threads.clamp(MIN_NUM_THREADS, MAX_NUM_THREADS);
    println!("\ncompute_betweenness: num_threads {:?}", num_threads);

    let num_nodes = indices.len();

    let mut betweenness_count: Vec<f64> = vec![0.0; num_nodes];

    let mut handles = Vec::with_capacity(num_threads);
    let wrapped_indices = Arc::new(indices);
    let wrapped_counter = Arc::new(Mutex::new(0));

    for _ in 0..num_threads {
        let acounter = Arc::clone(&wrapped_counter);
        let aindices = Arc::clone(&wrapped_indices);
        let handle = thread::spawn(move || betweenness_task(acounter, aindices));
        handles.push(handle);
    }

    let divisor: f64 = if normalize {
        ((num_nodes - 1) * (num_nodes - 2)) as f64
    } else {
        // non-normalized: everything is counted twice, so we must divide by two
        2.0
    };
    for h in handles {
        let b = h.join().unwrap();
        for i in 0..num_nodes {
            betweenness_count[i] += b[i] / divisor;
        }
    }

    println!("compute_betweenness: done {:?}", start.elapsed());

    betweenness_count
}
