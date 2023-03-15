use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

fn betweenness_for_node(
    index: usize,
    indices: &Vec<Vec<usize>>,
    betweenness_count: &mut Vec<u32>,
    total_path_length: &mut Vec<u32>,
    num_paths: &mut Vec<u32>,
) {
    let num_nodes = indices.len();
    let mut visited: Vec<bool> = vec![false; num_nodes];
    let mut search_state: Vec<bool> = vec![false; num_nodes];

    // mark node index and all those before index as searched, this sets
    // up the search space for the next iterations of the loop.
    for search_state in search_state.iter_mut().take(index + 1) {
        *search_state = true;
    }

    let mut search_list: Vec<usize> = Vec::with_capacity(num_nodes - index - 1);
    // we are searching for all j's that are greater than index
    for j in index + 1..num_nodes {
        search_list.push(j);
    }

    while !search_list.is_empty() {
        // 0. OUR MAIN SEARCH LOOP:  I and J
        // 1. we search for path between i and j.  We're done when we find j
        // 2. any short paths we find along the way, they get handled and removed from search list
        // 3. along the way, we appropriately mark any between nodes
        // 4. we also determine if no path exists (disconnected graph case)
        let mut done = false;
        let j = search_list[0];
        for (x, visited) in visited.iter_mut().enumerate().take(num_nodes) {
            *visited = x == index;
        }
        let mut pathlen: u32 = 1;
        let path = vec![index];
        let mut path_list = Vec::new();
        path_list.push(path);

        while !done {
            // for all shortest paths we find (and not necessily the i-j path we
            // are currently searching for), we store all of them here. And for one
            // node (i-j, or i-p, i-q...) there may be muliple paths that are shortest
            // and have same end points.
            let mut found_for_this_pathlen: Vec<usize> = Vec::new();
            // this list store the next unvisited node, to be
            // used as a starting node in the next round
            let mut queued_for_next_round = Vec::new();
            let mut touched: bool = false;
            for path in path_list.as_slice() {
                let q = path[path.len() - 1];
                let vertex = &indices[q];
                for x in vertex {
                    // Check if we've been here before
                    if !visited[*x] {
                        // if not, we're still not necessarily disconnected for this i-j instance
                        touched = true;
                        // one of our starting nodes for next round
                        let newpath = [path.as_slice(), &[*x]].concat();
                        if !search_state[*x] {
                            // if this i-x is to be searched, then we're done for that pair
                            // but we queue it first, in case other paths for same i-q are found
                            found_for_this_pathlen.push(*x);
                            if newpath.len() > 2 {
                                for i in 1..newpath.len() - 1 {
                                    let index = newpath[i];
                                    betweenness_count[index] += 1;
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
                let index = path[path.len() - 1];
                path_list.push(path.clone());
                visited[index] = true;
            }
            // now we do bookkeeping for any found
            // shortest paths.
            for f in found_for_this_pathlen {
                num_paths[f] += 1;
                total_path_length[f] += pathlen;
                num_paths[index] += 1;
                total_path_length[index] += pathlen;
                search_list.retain(|&x| x != f);
                search_state[f] = true;
                if f == j {
                    done = true;
                }
            }
            // If no connection exists, stop searching for it.
            if !touched {
                search_list.retain(|&x| x != j);
                search_state[j] = true;
                done = true
            }

            pathlen += 1;
        }
    }
}

pub fn betweenness_task(
    acounter: Arc<Mutex<usize>>,
    aindices: Arc<Vec<Vec<usize>>>,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let start = Instant::now();
    let indices = &aindices;
    let num_nodes = indices.len();
    // each worker thread keeps it own lists of data
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
