use std::{
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use crate::graph::GraphIndex;

const MIN_NUM_THREADS: usize = 1;
const MAX_NUM_THREADS: usize = 128;

/// This is a BFS, Breadth First Search implementation
/// In addition to counting betweenness attributes, path
/// lengths and counts are recorded, for quick access
/// to closeness centrality data
fn betweenness_for_node(
    index: usize,
    indices: &Vec<Vec<GraphIndex>>,
    betweenness_count: &mut [u32],
    total_path_length: &mut [u32],
    num_paths: &mut [u32],
) {
    let num_nodes = indices.len();
    let mut search_state: Vec<bool> = vec![false; num_nodes];

    // mark node index and all those before index as searched, this sets
    // up the search space for the next iterations of the loop.
    for search_state in search_state.iter_mut().take(index + 1) {
        *search_state = true;
    }

    let mut search_list: Vec<GraphIndex> = Vec::with_capacity(num_nodes - index - 1);
    // we are searching for all j's that are greater than index
    for j in index + 1..num_nodes {
        search_list.push(j as GraphIndex);
    }

    while !search_list.is_empty() {
        // 0. OUR MAIN SEARCH LOOP: for all j's greater than index (the node we're handling)
        // 1. we search for all the shortest paths between index and j.  We're done when we find j
        // 2. any short paths we find along the way, they get handled and removed from search list
        // 3. along the way, we appropriately mark any between nodes in shortest paths.
        // 4. we also determine if no path exists (disconnected graph case)

        // grab our next j
        let j = search_list[0];

        // we need to track visited status for ALL nodes (except for index)
        let mut visited: Vec<bool> = vec![false; num_nodes];
        visited[index] = true;

        // every journey begins with a single step
        let mut pathlen: u32 = 1;

        // initialize (seed) first path with our index node
        let path = vec![index as GraphIndex];
        let mut path_list = Vec::new();
        path_list.push(path);

        let mut done = false;
        while !done {
            // for all shortest paths we find (and not necessily the i-j path we
            // are currently searching for), we store all of them here. And for one
            // node (i-j, or i-p, i-q...) there may be muliple distince paths that
            // are shortest and have same end points.
            let mut found_for_this_pathlen: Vec<GraphIndex> = Vec::new();

            // this list stores the next unvisited node index,
            // to be used as a starting node in the next round
            let mut queued_for_next_round = Vec::new();

            let mut touched: bool = false;
            for path in path_list.as_slice() {
                let q = path[path.len() - 1];
                let vertex = &indices[q as usize];
                for x in vertex {
                    // Check if we've been here before
                    if !visited[*x as usize] {
                        // if not, we're still not necessarily disconnected for this i-j instance
                        touched = true;
                        // one of our starting nodes for next round
                        let newpath = [path.as_slice(), &[*x]].concat();
                        if !search_state[*x as usize] {
                            // if this i-x is to be searched, then we're done for that pair
                            // but we queue it first, in case other paths for same i-q are found
                            found_for_this_pathlen.push(*x);
                            if newpath.len() > 2 {
                                // Now we can increment the betweenness counts: newpath is a Shortest Path
                                // Of course, we skip the first and last nodes
                                for b in newpath.iter().take(newpath.len() - 1).skip(1) {
                                    betweenness_count[*b as usize] += 1;
                                }
                            }
                        }
                        queued_for_next_round.push(newpath);
                    }
                }
            }

            // prep for next round, start fresh queue list
            path_list.clear();
            // load up the queue list, marked as visited
            for path in queued_for_next_round {
                let end_index = path[path.len() - 1];
                path_list.push(path);
                visited[end_index as usize] = true;
            }

            // now we do bookkeeping for any found
            // shortest paths.
            for f in found_for_this_pathlen {
                num_paths[f as usize] += 1;
                num_paths[index] += 1;
                total_path_length[f as usize] += pathlen;
                total_path_length[index] += pathlen;
                search_list.retain(|&x| x != f);
                search_state[f as usize] = true;
                if f == j {
                    done = true;
                }
            }

            // If no connection exists, stop searching for it.
            if !touched {
                search_list.retain(|&x| x != j);
                search_state[j as usize] = true;
                done = true
            }

            pathlen += 1;
        }
    }
}


/// this function is the thread task
/// grabs next unprocessed node
/// if no more nodes, exits
/// returning betweenness, total path lengths, and num paths
fn betweenness_task(
    acounter: Arc<Mutex<usize>>,
    aindices: Arc<Vec<Vec<GraphIndex>>>,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let start = Instant::now();
    let indices = &aindices;
    let num_nodes = indices.len();

    // each worker thread keeps its own cache of data
    // these are returned when the thread finishes
    // and then summed by the caller
    let mut betweenness_count: Vec<u32> = vec![0; num_nodes];
    let mut total_path_length: Vec<u32> = vec![0; num_nodes];
    let mut num_paths: Vec<u32> = vec![0; num_nodes];

    let mut finished = false;
    while !finished {
        let mut counter = acounter.lock().unwrap();
        let index: usize = *counter;
        *counter += 1;
        drop(counter);
        if index < num_nodes - 1 {
            if index % 100 == 0 {
                println!("node: {}, time: {:?}", index, start.elapsed());
            }
            betweenness_for_node(
                index,
                indices,
                &mut betweenness_count,
                &mut total_path_length,
                &mut num_paths,
            );
        } else {
            finished = true;
        }
    }
    (betweenness_count, total_path_length, num_paths)
}

pub fn compute_betweenness(
    indices: Vec<Vec<GraphIndex>>,
    mut num_threads: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let start = Instant::now();
    num_threads = num_threads.clamp(MIN_NUM_THREADS, MAX_NUM_THREADS);
    println!("\ncompute: num_threads {:?}", num_threads);

    let num_nodes = indices.len();

    let mut betweenness_count: Vec<u32> = vec![0; num_nodes];
    let mut total_path_length: Vec<u32> = vec![0; num_nodes];
    let mut num_paths: Vec<u32> = vec![0; num_nodes];

    let mut handles = Vec::new();
    let wrapped_indices = Arc::new(indices);
    let wrapped_counter = Arc::new(Mutex::new(0));

    for _ in 0..num_threads {
        let acounter = Arc::clone(&wrapped_counter);
        let aindices = Arc::clone(&wrapped_indices);
        let handle = thread::spawn(move || betweenness_task(acounter, aindices));
        handles.push(handle);
    }

    for h in handles {
        let (b, t, n) = h.join().unwrap();
        for i in 0..num_nodes {
            betweenness_count[i] += b[i];
            total_path_length[i] += t[i];
            num_paths[i] += n[i];
        }
    }

    println!("compute: done {:?}", start.elapsed());

    (betweenness_count, total_path_length, num_paths)
}
