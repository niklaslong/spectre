//! A module for working with graphs.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::Hash,
    ops::Sub,
    time::{Instant},
    thread,
    sync::{Arc, Mutex},
};

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::edge::Edge;
use crate::compute::{betweenness_task, S};

// struct S {
//     indices: Vec<Vec<usize>>,
// }

// //     pub indices: &Vec<Vec<usize>>,
// //     // pub start: Instant,
// //     // pub last_elapsed: Duration,
// //     pub betweenness_count: &Vec<u32>,
// //     pub total_path_length: &Vec<u32>,
// //     pub num_paths: &Vec<u32>,
// //     pub next_index: usize,
// fn betweenness_for_node( 
//     index: usize, 
//     indices: &Vec<Vec<usize>>,
//     betweenness_count: &mut Vec<u32>,
//     total_path_length: &mut Vec<u32>,
//     num_paths: &mut Vec<u32>,

// ) {
//     // let state = self.state.unwrap();
//     // let indices = s.indices;
//     let num_nodes = indices.len();
//     let mut visited: Vec<bool> = vec![false; num_nodes];
//     let mut search_state: Vec<bool> = vec![false; num_nodes];
//     // let _elapsed = start.elapsed();
//     // println!("  node: {:?}, {:?}, delta {:}", index, elapsed, elapsed.as_secs_f64() - state.last_elapsed.as_secs_f64());
//     // last_elapsed = elapsed;

//     // mark node i and all those before i as searched, this sets
//     // up the search space for the next iterations of the loop.
//     for search_state in search_state.iter_mut().take(index + 1) {
//         *search_state = true;
//     }

//     let mut search_list: Vec<usize> = Vec::with_capacity(num_nodes - index - 1);
//     for j in index+1..num_nodes {
//         search_list.push(j);
//     }

//     while !search_list.is_empty() {
//         // 0. OUR MAIN SEARCH LOOP:  I and J
//         // 1. we search for path between i and j.  We're done when we find j
//         // 2. any short paths we find along the way, they get handled and removed from search list
//         // 3. along the way, we appropriately mark any between nodes
//         // 4. we also determine if no path exists (disconnected graph case)
//         let mut done = false;
//         let j = search_list[0];
//         for (x, visited) in visited.iter_mut().enumerate().take(num_nodes) {
//             *visited = x == index;
//         }
//         let mut pathlen: u32 = 1;
//         let path = vec![index];
//         let mut path_list = Vec::new();
//         path_list.push(path);

//         while !done {
//             // for all shortest paths we find (and not necessily the i-j path we
//             // are currently searching for), we store all of them here. And for one
//             // node (i-j, or i-p, i-q...) there may be muliple paths that are shortest
//             // and have same end points.
//             let mut found_for_this_pathlen: Vec<usize> = Vec::new();
//             // this list store the next unvisited node, to be
//             // used as a starting node in the next round
//             let mut queued_for_next_round = Vec::new();
//             let mut touched: bool = false;
//             for path in path_list.as_slice() {
//                 let q = path[path.len() - 1];
//                 let vertex = &indices[q];
//                 for x in vertex {
//                     // Check if we've been here before
//                     if !visited[*x] {
//                         // if not, we're still not necessarily disconnected for this i-j instance
//                         touched = true;
//                         // one of our starting nodes for next round
//                         let mut newpath = path.clone();
//                         newpath.push(*x);
//                         if !search_state[*x] {
//                             // if this i-x is to be searched, then we're done for that pair
//                             // but we queue it first, in case other paths for same i-q are found
//                             found_for_this_pathlen.push(*x);
//                             if newpath.len() > 2 {
//                                 for i in 1..newpath.len() - 1 {
//                                     let index = newpath.get(i).unwrap();
//                                     betweenness_count[*index] += 1;
//                                 }
//                             }
//                         }
//                         queued_for_next_round.push(newpath);
//                     }
//                 }
//             }

//             // prep for next round, start fresh queue list
//             path_list.clear();
//             // load up the queue list, marked as visited
//             for path in queued_for_next_round {
//                 let index = path[path.len() - 1];
//                 path_list.push(path.clone());
//                 visited[index] = true;
//             }
//             // now we do bookkeeping for any found
//             // shortest paths.
//             for f in found_for_this_pathlen {
//                 num_paths[f] += 1;
//                 total_path_length[f] += pathlen;
//                 num_paths[index] += 1;
//                 total_path_length[index] += pathlen;
//                 search_list.retain(|&x| x != f);
//                 search_state[f] = true;
//                 if f == j {
//                     done = true;
//                 }
//             }
//             // If no connection exists, stop searching for it.
//             if !touched {
//                 search_list.retain(|&x| x != j);
//                 search_state[j] = true;
//                 done = true
//             }

//             pathlen += 1;
//         }
//     }
// }

// fn betweenness_task(
//    s: &S,
// //    indices: Vec<Vec<usize>>,
//     c: Arc<Mutex<usize>>,
//     // start_index: usize,
//     // end_index: usize,
// ) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
//     let start = Instant::now();
//     // println!("task here, start_index {}, end_index {}", start_index, end_index);
//     // let elapsed = start.elapsed();
//     // println!("compute A start {:?}: {:?}", start_index, start.elapsed());

//     let indices = &s.indices;
//     let num_nodes = indices.len();
//     let mut betweenness_count: Vec<u32> = vec![0; num_nodes];
//     let mut total_path_length: Vec<u32> = vec![0; num_nodes];
//     let mut num_paths: Vec<u32> = vec![0; num_nodes];
//     let mut finished = false;
//     while !finished {
//         let mut counter = c.lock().unwrap();
//         let index: usize = *counter;
//         *counter += 1;
//         drop(counter);
//         if index < num_nodes - 1 {
//             if index % 100 == 0 {
//                 println!("node: {}, time: {:?}", index, start.elapsed());
//             }
//             betweenness_for_node(index, indices, &mut betweenness_count, &mut total_path_length, &mut num_paths);
//         } else {
//             finished = true;
//         }
//     }
//     (betweenness_count, total_path_length, num_paths)
// }




/// An undirected graph, made up of edges.
#[derive(Clone, Debug)]
pub struct Graph<T> {
    /// The edges in the graph.
    edges: HashSet<Edge<T>>,
    /// A mapping of vertices to their indices to be used when constructing the various matrices
    /// representing the graph.
    ///
    /// The use of a `BTreeMap` means we need the `Ord` bound on `T`. The sorted collection allows
    /// us to maintain some form of order between computations, which can be useful for debugging.
    index: Option<BTreeMap<T, usize>>,
    /// Cache the degree matrix when possible.
    degree_matrix: Option<DMatrix<f64>>,
    /// Cache the adjacency matrix when possible.
    adjacency_matrix: Option<DMatrix<f64>>,
    /// Cache the laplacian matrix when possible.
    laplacian_matrix: Option<DMatrix<f64>>,
    /// Cache the betweenness count when possible.
    betweenness_count: Option<Vec<u32>>,
    /// Cache the path lengths when possible.
    total_path_length: Option<Vec<u32>>,
    /// Cache the num paths when possible.
    num_paths: Option<Vec<u32>>,

    // pub indices: Vec<Vec<usize>>,
    // pub betweennesscount: Vec<u32>,
    // pub totalpathlength: Vec<u32>,
    // pub numpaths: Vec<u32>,
    // pub start: Instant,


    // state: Option<State>,


}

impl<T> Default for Graph<T>
where
    Edge<T>: Eq + Hash,
    T: Copy + Eq + Hash + Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Graph<T>
where
    Edge<T>: Eq + Hash,
    T: Copy + Eq + Hash + Ord,
{
    /// Creates an empty graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectre::graph::Graph;
    ///
    /// let graph: Graph<&str> = Graph::new();
    /// ```
    pub fn new() -> Self {
        Self {
            edges: Default::default(),
            index: None,
            degree_matrix: None,
            adjacency_matrix: None,
            laplacian_matrix: None,
            betweenness_count: None,
            total_path_length: None,
            num_paths: None,
            // indices: Vec::new(),
            // betweennesscount: Vec::new(),
            // totalpathlength: Vec::new(),
            // numpaths: Vec::new(),
            // start: Instant::now(),
        
            // state: None,
        }
    }

    pub fn edges(&mut self) -> &HashSet<Edge<T>> {
        &self.edges
    }

    /// Inserts an edge into the graph.
    pub fn insert(&mut self, edge: Edge<T>) -> bool {
        let is_inserted = self.edges.insert(edge);

        // Delete the cached objects if the edge was successfully inserted because we can't
        // reliably update them from the new connection alone.
        if is_inserted && self.index.is_some() {
            self.clear_cache()
        }

        is_inserted
    }

    /// Inserts a subset of `(hub, leaf)` edges into the graph.
    pub fn insert_subset(&mut self, hub: T, leaves: &[T]) {
        for leaf in leaves {
            self.insert(Edge::new(hub, *leaf));
        }
    }

    /// Inserts a subset of `(hub, leaf)` edges into the graph and removes any existing edges that
    /// contain the hub but aren't included in the new set.
    pub fn update_subset(&mut self, hub: T, leaves: &[T]) {
        let new_edges: HashSet<Edge<T>> = leaves.iter().map(|leaf| Edge::new(hub, *leaf)).collect();

        // Remove hub-containing edges that aren't included in the new set.
        let original_len = self.edge_count();
        self.edges
            .retain(|edge| new_edges.contains(edge) || !edge.contains(&hub));

        // Make sure to clear the cache after removals as there may be no inserts.
        // TODO: make more efficient.
        if self.edge_count() != original_len {
            self.clear_cache()
        }

        for edge in new_edges {
            self.insert(edge);
        }
    }

    /// Removes an edge from the set and returns whether it was present in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    ///
    /// assert_eq!(graph.remove(&Edge::new("a", "b")), true);
    /// assert_eq!(graph.remove(&Edge::new("a", "c")), false);
    /// ```
    pub fn remove(&mut self, edge: &Edge<T>) -> bool {
        let is_removed = self.edges.remove(edge);

        // Delete the cached objects if the edge was successfully removed because we can't reliably
        // update them from the new connection alone.
        if is_removed && self.index.is_some() {
            self.clear_cache()
        }

        is_removed
    }

    /// Checks if the graph contains an edge.
    pub fn contains(&self, edge: &Edge<T>) -> bool {
        self.edges.contains(edge)
    }

    /// Returns the vertex count of the graph.
    ///
    /// This call constructs the collection of vertices from the collection of edges. This is
    /// because the vertex set can't accurately be updated on the basis of the addition or the
    /// removal of an edge alone.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    ///
    /// assert_eq!(graph.vertex_count(), 2);
    /// ```
    pub fn vertex_count(&self) -> usize {
        self.vertices_from_edges().len()
    }

    /// Returns the edge count of the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Computes the density of the graph, the ratio of edges with respect to the maximum possible
    /// edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    ///
    /// graph.insert(Edge::new("a", "b"));
    /// assert_eq!(graph.density(), 1.0);
    ///
    /// graph.insert(Edge::new("a", "c"));
    /// assert_eq!(graph.density(), 2.0 / 3.0);
    /// ```
    pub fn density(&self) -> f64 {
        let vc = self.vertex_count() as f64;
        let ec = self.edge_count() as f64;

        // Calculate the total number of possible edges given a vertex count.
        let pec = vc * (vc - 1.0) / 2.0;
        // Actual edges divided by the possible edges gives the density.
        ec / pec
    }

    /// Constructs the adjacency matrix for this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::dmatrix;
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    /// assert_eq!(
    ///     graph.adjacency_matrix(),
    ///     dmatrix![0.0, 1.0;
    ///              1.0, 0.0]
    /// );
    /// ```
    pub fn adjacency_matrix(&mut self) -> DMatrix<f64> {
        // Check the cache.
        if let Some(matrix) = self.adjacency_matrix.clone() {
            return matrix;
        }

        self.generate_index();

        // Safety: the previous call guarantees the index has been generated and stored.
        let n = self.index.as_ref().unwrap().len();
        let mut matrix = DMatrix::<f64>::zeros(n, n);

        // Compute the adjacency matrix. As our we're assuming the graph is undirected, the adjacency matrix is
        // symmetric.
        for edge in &self.edges {
            // Safety: get the indices for each edge in the graph, these must be present as the
            // index was generated from this set of edges.
            let i = self.index.as_ref().unwrap().get(edge.source()).unwrap();
            let j = self.index.as_ref().unwrap().get(edge.target()).unwrap();

            // Since edges are guaranteed to be unique, both the upper and lower triangles must be
            // writted (as the graph is unidrected) for each edge.
            matrix[(*i, *j)] = 1.0;
            matrix[(*j, *i)] = 1.0;
        }

        // Cache the matrix.
        self.adjacency_matrix = Some(matrix.clone());

        matrix
    }

    /// Constructs the degree matrix for this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::dmatrix;
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    /// assert_eq!(
    ///     graph.degree_matrix(),
    ///     dmatrix![1.0, 0.0;
    ///              0.0, 1.0]
    /// );
    /// ```
    pub fn degree_matrix(&mut self) -> DMatrix<f64> {
        // Check the cache.
        if let Some(matrix) = self.degree_matrix.clone() {
            return matrix;
        }

        let adjacency_matrix = self.adjacency_matrix();

        // Safety: the previous call guarantees the index has been generated and stored.
        let n = self.index.as_ref().unwrap().len();
        let mut matrix = DMatrix::<f64>::zeros(n, n);

        for (i, row) in adjacency_matrix.row_iter().enumerate() {
            // Set the diagonal to be the sum of edges in that row. The index isn't necessary
            // here since the rows are visited in order and the adjacency matrix is ordered after the
            // index.
            matrix[(i, i)] = row.sum()
        }

        // Cache the matrix.
        self.degree_matrix = Some(matrix.clone());

        matrix
    }

    /// Constructs the laplacian matrix for this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::dmatrix;
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    /// assert_eq!(
    ///     graph.laplacian_matrix(),
    ///     dmatrix![1.0, -1.0;
    ///              -1.0, 1.0]
    /// );
    /// ```
    pub fn laplacian_matrix(&mut self) -> DMatrix<f64> {
        // Check the cache.
        if let Some(matrix) = self.laplacian_matrix.clone() {
            return matrix;
        }

        let degree_matrix = self.degree_matrix();
        let adjacency_matrix = self.adjacency_matrix();

        let matrix = degree_matrix.sub(&adjacency_matrix);

        // Cache the matrix.
        self.laplacian_matrix = Some(matrix.clone());

        matrix
    }

    /// Returns the difference between the highest and lowest degree centrality in the network.
    ///
    /// Returns an `f64`, though the value should be a natural number.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    /// graph.insert(Edge::new("a", "c"));
    ///
    /// assert_eq!(graph.degree_centrality_delta(), 1.0);
    /// ```
    pub fn degree_centrality_delta(&mut self) -> f64 {
        let degree_matrix = self.degree_matrix();

        let max = degree_matrix.diagonal().max();
        let min = degree_matrix.diagonal().min();

        max - min
    }

    /// Returns a mapping of vertices to their degree centrality (number of connections) in the graph.
    pub fn degree_centrality(&mut self) -> HashMap<T, u32> {
        let degree_matrix = self.degree_matrix();

        // Safety: the previous call guarantees the index has been generated and stored.
        self.index
            .as_ref()
            .unwrap()
            .keys()
            .zip(degree_matrix.diagonal().iter())
            .map(|(addr, dc)| (*addr, *dc as u32))
            .collect()
    }

    /// Returns a mapping of vertices to their eigenvalue centrality (the relative importance of
    /// the vertex) in the graph.
    pub fn eigenvalue_centrality(&mut self) -> HashMap<T, f64> {
        let adjacency_matrix = self.adjacency_matrix();

        // Early return if the matrix is empty, the rest of the computation requires a matrix with
        // at least a dim of 1x1.
        if adjacency_matrix.is_empty() {
            return HashMap::new();
        }

        // Compute the eigenvectors and corresponding eigenvalues and sort in descending order.
        let ascending = false;
        let eigenvalue_vector_pairs = sorted_eigenvalue_vector_pairs(adjacency_matrix, ascending);
        let (_highest_eigenvalue, highest_eigenvector) = &eigenvalue_vector_pairs[0];

        // The eigenvector is a relative score of vertex importance (normalised by the norm), to obtain an absolute score for each
        // vertex, we normalise so that the sum of the components are equal to 1.
        let sum = highest_eigenvector.sum() / self.index.as_ref().unwrap().len() as f64;
        let normalised = highest_eigenvector.unscale(sum);

        // Map addresses to their eigenvalue centrality.
        self.index
            .as_ref()
            .unwrap()
            .keys()
            .zip(normalised.column(0).iter())
            .map(|(addr, ec)| (*addr, *ec))
            .collect()
    }

    /// Returns the algebraic connectivity (Fiedler eigenvalue) of the graph and a mapping of the
    /// vertices to their Fiedler value (their associated component in the Fiedler eigenvector).
    pub fn fiedler(&mut self) -> (f64, HashMap<T, f64>) {
        let laplacian_matrix = self.laplacian_matrix();

        // Early return if the matrix is empty, the rest of the computation requires a matrix with
        // at least a dim of 1x1.
        if laplacian_matrix.is_empty() {
            return (0.0, HashMap::new());
        }

        // Compute the eigenvectors and corresponding eigenvalues and sort in ascending order.
        let ascending = true;
        let pairs = sorted_eigenvalue_vector_pairs(laplacian_matrix, ascending);

        // Second-smallest eigenvalue of the Laplacian is the Fiedler value (algebraic connectivity), the associated
        // eigenvector is the Fiedler vector.
        let (algebraic_connectivity, fiedler_vector) = &pairs[1];

        // Map addresses to their Fiedler values.
        let fiedler_values_indexed = self
            .index
            .as_ref()
            .unwrap()
            .keys()
            .zip(fiedler_vector.column(0).iter())
            .map(|(addr, fiedler_value)| (*addr, *fiedler_value))
            .collect();

        (*algebraic_connectivity, fiedler_values_indexed)
    }

    //
    // Private
    //

    /// Clears the computed state.
    ///
    /// This should be called every time the set of edges is mutated since the cached state won't
    /// correspond to the new graph.
    fn clear_cache(&mut self) {
        self.index = None;
        self.degree_matrix = None;
        self.adjacency_matrix = None;
        self.laplacian_matrix = None;
        self.betweenness_count = None;
        self.total_path_length = None;
        self.num_paths = None;
    }

    /// Returns the set of unique vertices contained within the set of edges.
    fn vertices_from_edges(&self) -> HashSet<T> {
        let mut vertices: HashSet<T> = HashSet::new();
        for edge in self.edges.iter() {
            // Using a hashset guarantees uniqueness.
            vertices.insert(*edge.source());
            vertices.insert(*edge.target());
        }

        vertices
    }

    /// Constructs and stores an index of vertices for this set of edges.
    ///
    /// The index will be sorted by `T`'s implementation of `Ord`.
    fn generate_index(&mut self) {
        // It should be impossible to call this function if the cache is not empty.
        debug_assert!(self.index.is_none());

        let mut vertices: Vec<T> = self.vertices_from_edges().into_iter().collect();
        vertices.sort();

        let index: BTreeMap<T, usize> = vertices
            .iter()
            .enumerate()
            .map(|(i, &vertex)| (vertex, i))
            .collect();

        self.index = Some(index);
    }

    /// This method returns a set connection indices for each node.
    /// It a compact way to view the adjacency matrix, and therefore, is
    /// used for the computation of betweenness and closeness centralities
    pub fn get_adjacency_indices(&mut self) -> Vec<Vec<usize>> {
        let mut indices: Vec<Vec<usize>> = Vec::new();
        let adjacency_matrix = self.adjacency_matrix();

        for m in 0..adjacency_matrix.nrows() {
            let neighbors: Vec<usize> = adjacency_matrix
                .row(m)
                .iter()
                .enumerate()
                .filter(|(_n, &val)| val == 1.0)
                .map(|(n, _)| n)
                .collect();
            indices.push(neighbors);
        }
        indices
    }

    /// This method also outputs an array of index vectors, although it is created differently.
    /// It is currently used if filtering of nodes is required.
    pub fn get_filtered_adjacency_indices(&self, nodes_to_keep: &Vec<T>) -> Vec<Vec<usize>> {
        let num_nodes = nodes_to_keep.len();
        let mut indices = Vec::with_capacity(num_nodes);
        let mut node_map = HashMap::with_capacity(num_nodes);
        for n in 0..num_nodes {
            // make initial capacity 10% of total
            indices.push(Vec::with_capacity(num_nodes / 10));
            node_map.insert(nodes_to_keep[n], n);
        }

        // For each edge, check if the source and target nodes
        // are in our node HashMap.  If we've obtained both
        // indices, insert into the corresponding connection list
        for edge in self.edges.iter() {
            if let Some(source_index) = node_map.get(edge.source()) {
                if let Some(target_index) = node_map.get(edge.target()) {
                    indices[*source_index].push(*target_index);
                    indices[*target_index].push(*source_index);
                }
            }
        }

        for node in indices.iter_mut() {
            node.shrink_to_fit();
        }

        indices
    }
    


    /// This method computes the closeness and betweenness for a given Graph.
    ///
    /// Closeness: for each node, find all shortest paths to all other nodes.
    /// Accumulate all path lengths, accumulate number of paths, and then compute
    /// average path length.
    ///
    /// Betweenness: When a shortest path is found, for all nodes
    /// in-between (i.e., not an end point), increment their betweenness value.
    /// Normalize the counts by dividing by the number of shortest paths found
    ///
    fn betweenness_and_closeness_centrality(&mut self, num_threads: usize) {
        let start = Instant::now();
        let elapsed = start.elapsed();
        println!("\ncompute: num_threads {:?}", num_threads);

        if self.betweenness_count.is_some() {
            return;
        }
        let indices: Vec<Vec<usize>> = self.get_adjacency_indices();
        let num_nodes = indices.len();

        println!("compute: B {:?}", start.elapsed());
        let mut betweenness_count: Vec<u32> = vec![0; num_nodes];
        let mut total_path_length: Vec<u32> = vec![0; num_nodes];
        let mut num_paths: Vec<u32> = vec![0; num_nodes];

        // the last searchable pair is:
        //     i = num_nodes - 2
        //     j = num_nodes - 1
        // let last_elapsed = start.elapsed();
        // let mut state = State {
        //     &indices, 
        //     start,
        //     last_elapsed: start.elapsed(),
        //     betweenness_count: &betweenness_count,
        //     total_path_length: &total_path_length, 
        //     num_paths: &num_paths,
        //     next_index: 0,
        // };

        // doit(&mut state);
        let mut handles = Vec::new();
        // let num_threads = 2;
        // let mut start_indices = Vec::new();
        // println!("num_nodes {}", num_nodes);
        // for t in 0..num_threads+1 {
        //     let part = (num_threads - t) as f64 / num_threads as f64;
        //     let section = 1.0 - part.powf(1.0/1.6);
        //     let mut index = (section * num_nodes as f64).floor() as usize;
        //     if index > num_nodes - 1 {
        //         index = num_nodes - 1;
        //     }
        //     println!("t:{t}, part:{part}, section:{section}, index:{index}");
        //     start_indices.push(index);

        // }
        let s = Arc::new(S { indices });
        let counter = Arc::new(Mutex::new(0 as usize));
        for t in 0..num_threads {
            let ss = s.clone();
            let cc = Arc::clone(&counter);
            // let ii = indices.clone();
            // let start_index = start_indices[t];
            // let end_index = start_indices[t+1];
            //if start_index < end_index {
                let handle = thread::spawn(move || {
                    betweenness_task(
                        // ii,
                        &ss,
                        cc,
                        // end_index,
                        // &start
                    )
                });    
                handles.push(handle);
            // }
        }
        //let h = handles[0];
        for h in handles {
            let (b, t, n) = h.join().unwrap();
            for i in 0..num_nodes {
                betweenness_count[i] += b[i];
                total_path_length[i] += t[i];
                num_paths[i] += n[i];
            }
            println!("thread done ");
        }
        // for i in 0..num_nodes - 1 {
        //     self.betweenness_for_node(i, &mut state);

        // }
        println!("compute: C {:?}", start.elapsed());

        self.betweenness_count = Some(betweenness_count);
        self.total_path_length = Some(total_path_length);
        self.num_paths = Some(num_paths);
        // self.betweenness_count = Some(b);
        // self.total_path_length = Some(t);
        // self.num_paths = Some(n);
    }

    /// This method returns the betweenness for a given Graph.
    ///
    /// Betweenness: When a shortest path is found, for all nodes
    /// in-between (i.e., not an end point), increment their betweenness value.
    /// Normalize the counts by dividing by the number of shortest paths found
    ///
    pub fn betweenness_centrality(&mut self, num_threads: usize) -> HashMap<T, f64> {
        self.betweenness_and_closeness_centrality(num_threads);

        let betweenness_count = self.betweenness_count.as_ref().unwrap();
        let num_paths = self.num_paths.as_ref().unwrap();

        let mut total_num_paths: u32 = 0;
        for num_paths in num_paths.iter() {
            total_num_paths += num_paths;
        }

        let mut centralities = HashMap::new();
        for (node, i) in self.index.as_ref().unwrap() {
            let value = betweenness_count[*i] as f64 / total_num_paths as f64;
            centralities.insert(*node, value);
        }
        centralities
    }

    /// This method returns the closeness for a given Graph.
    ///
    /// Closeness: for each node, find all shortest paths to all other nodes.
    /// Accumulate all path lengths, accumulate number of paths, and then compute
    /// average path length.
    pub fn closeness_centrality(&mut self, num_threads: usize) -> HashMap<T, f64> {
        self.betweenness_and_closeness_centrality(num_threads);

        let total_path_length = self.total_path_length.as_ref().unwrap();
        let num_paths = self.num_paths.as_ref().unwrap();

        let mut total_num_paths: u32 = 0;
        for num_paths in num_paths.iter() {
            total_num_paths += num_paths;
        }

        let mut centralities = HashMap::new();
        for (n, node) in self.index.as_ref().unwrap().keys().enumerate() {
            let value = total_path_length[n] as f64 / num_paths[n] as f64;
            centralities.insert(*node, value);
        }

        centralities
    }
}

//
// Helpers
//

/// Computes the eigenvalues and corresponding eigenvalues from the supplied symmetric matrix.
fn sorted_eigenvalue_vector_pairs(
    matrix: DMatrix<f64>,
    ascending: bool,
) -> Vec<(f64, DVector<f64>)> {
    // Early return if the matrix is empty, the rest of the computation requires a matrix with
    // at least a dim of 1x1.
    if matrix.is_empty() {
        return vec![];
    }

    // Compute eigenvalues and eigenvectors.
    let eigen = SymmetricEigen::new(matrix);

    // Map eigenvalues to their eigenvectors.
    let mut pairs: Vec<(f64, DVector<f64>)> = eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
        .map(|(value, vector)| (*value, vector.clone_owned()))
        .collect();

    // Sort eigenvalue-vector pairs in descending order.
    pairs.sort_unstable_by(|(a, _), (b, _)| {
        if ascending {
            a.partial_cmp(b).unwrap()
        } else {
            b.partial_cmp(a).unwrap()
        }
    });

    pairs
}

#[cfg(test)]
mod tests {
    use std::fs;

    use nalgebra::dmatrix;
    use serde::Deserialize;

    use super::*;

    #[derive(Default, Clone, Deserialize, Debug)]
    pub struct Sample {
        pub node_ips: Vec<String>,
        pub indices: Vec<Vec<usize>>,
    }

    // Creates a graph from a list of paths (that can overlap, the graph handles deduplication).
    macro_rules! graph {
        ($($path:expr),*) => {{
            let mut graph = Graph::new();

            $(
                let mut iter = $path.into_iter().peekable();
                while let (Some(a), Some(b)) = (iter.next(), iter.peek()) {
                    graph.insert(Edge::new(a, b));
                }

            )*

            graph
        }}
    }

    #[test]
    fn new() {
        let _: Graph<()> = Graph::new();
    }

    #[test]
    fn insert() {
        let mut graph = Graph::new();
        let edge = Edge::new("a", "b");

        assert!(graph.insert(edge.clone()));
        assert!(!graph.insert(edge));
    }

    #[test]
    fn insert_subset() {
        let mut graph = Graph::new();

        let (a, b, c, d) = ("a", "b", "c", "d");

        graph.insert(Edge::new(a, b));
        graph.insert(Edge::new(a, c));

        let edges = vec![b, d];
        graph.insert_subset(a, &edges);

        assert!(graph.contains(&Edge::new(a, b)));
        assert!(graph.contains(&Edge::new(a, c)));
        assert!(graph.contains(&Edge::new(a, d)));

        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn update_subset() {
        let mut graph = Graph::new();

        let (a, b, c, d) = ("a", "b", "c", "d");

        graph.insert(Edge::new(a, b));
        graph.insert(Edge::new(a, c));
        graph.insert(Edge::new(b, c));

        let edges = vec![b, d];
        graph.update_subset(a, &edges);

        assert!(graph.contains(&Edge::new(a, b)));
        assert!(!graph.contains(&Edge::new(a, c)));
        assert!(graph.contains(&Edge::new(b, c)));
        assert!(graph.contains(&Edge::new(a, d)));

        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn remove() {
        let edge = Edge::new("a", "b");
        let uninserted_edge = Edge::new("a", "c");

        let mut graph = Graph::new();
        graph.insert(edge.clone());

        assert!(graph.remove(&edge));
        assert!(!graph.remove(&uninserted_edge));
    }

    #[test]
    fn contains() {
        let mut graph = Graph::new();
        let edge = Edge::new("a", "b");

        graph.insert(edge.clone());

        assert!(graph.contains(&edge));
        assert!(!graph.contains(&Edge::new("b", "c")));
    }

    #[test]
    fn vertex_count() {
        let mut graph = Graph::new();
        assert_eq!(graph.vertex_count(), 0);

        // Verify two new vertices get added when they don't yet exist in the graph.
        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.vertex_count(), 2);

        // Verify only one new vertex is added when one of them already exists in the graph.
        graph.insert(Edge::new("a", "c"));
        assert_eq!(graph.vertex_count(), 3);
    }

    #[test]
    fn edge_count() {
        let mut graph = Graph::new();
        assert_eq!(graph.edge_count(), 0);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn density() {
        let mut graph = Graph::new();
        assert!(graph.density().is_nan());

        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.density(), 1.0);

        graph.insert(Edge::new("a", "c"));
        assert_eq!(graph.density(), 2.0 / 3.0);
    }

    #[test]
    fn adjacency_matrix() {
        let mut graph = Graph::new();
        assert_eq!(graph.adjacency_matrix(), dmatrix![]);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(
            graph.adjacency_matrix(),
            dmatrix![0.0, 1.0;
                     1.0, 0.0]
        );

        graph.insert(Edge::new("a", "c"));
        assert_eq!(
            graph.adjacency_matrix(),
            dmatrix![0.0, 1.0, 1.0;
                     1.0, 0.0, 0.0;
                     1.0, 0.0, 0.0]
        );

        // Sanity check the index gets stored.
        assert!(graph.index.is_some());
    }

    #[test]
    fn degree_matrix() {
        let mut graph = Graph::new();
        assert_eq!(graph.degree_matrix(), dmatrix![]);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(
            graph.degree_matrix(),
            dmatrix![1.0, 0.0;
                     0.0, 1.0]
        );

        graph.insert(Edge::new("a", "c"));
        assert_eq!(
            graph.degree_matrix(),
            dmatrix![2.0, 0.0, 0.0;
                     0.0, 1.0, 0.0;
                     0.0, 0.0, 1.0]
        );

        // Sanity check the index gets stored.
        assert!(graph.index.is_some());
    }

    #[test]
    fn laplacian_matrix() {
        let mut graph = Graph::new();
        assert_eq!(graph.laplacian_matrix(), dmatrix![]);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(
            graph.laplacian_matrix(),
            dmatrix![1.0, -1.0;
                     -1.0, 1.0]
        );

        graph.insert(Edge::new("a", "c"));
        assert_eq!(
            graph.laplacian_matrix(),
            dmatrix![2.0, -1.0, -1.0;
                     -1.0, 1.0, 0.0;
                     -1.0, 0.0, 1.0]
        );

        // Sanity check the index gets stored.
        assert!(graph.index.is_some());
    }

    #[test]
    fn degree_centrality_delta() {
        let mut graph = Graph::new();
        assert_eq!(graph.degree_centrality_delta(), 0.0);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.degree_centrality_delta(), 0.0);

        graph.insert(Edge::new("a", "c"));
        assert_eq!(graph.degree_centrality_delta(), 1.0);
    }

    #[test]
    fn degree_centrality() {
        let mut graph = Graph::new();
        assert!(graph.degree_centrality().is_empty());

        // One connection, centrality measures for each vertex should be 1.
        let (a, b, c) = ("a", "b", "c");
        graph.insert(Edge::new(a, b));
        let degree_centrality = graph.degree_centrality();

        assert_eq!(degree_centrality.get_key_value(a), Some((&a, &1)));
        assert_eq!(degree_centrality.get_key_value(b), Some((&b, &1)));

        // Sanity check the length.
        assert_eq!(degree_centrality.len(), 2);

        // Two connections, degree centrality for A should increase.
        graph.insert(Edge::new(a, c));
        let degree_centrality = graph.degree_centrality();

        assert_eq!(degree_centrality.get_key_value(a), Some((&a, &2)));
        assert_eq!(degree_centrality.get_key_value(b), Some((&b, &1)));
        assert_eq!(degree_centrality.get_key_value(c), Some((&c, &1)));

        // Sanity check the length.
        assert_eq!(degree_centrality.len(), 3);
    }

    #[test]
    fn eigenvalue_centrality() {
        let mut graph = Graph::new();
        assert!(graph.eigenvalue_centrality().is_empty());

        // One connection, centrality measures for each vertex should be 1.
        let (a, b, c) = ("a", "b", "c");
        graph.insert(Edge::new(a, b));
        let eigenvalue_centrality = graph.eigenvalue_centrality();

        assert_eq!(eigenvalue_centrality.get_key_value(a), Some((&a, &1.0)));
        assert_eq!(eigenvalue_centrality.get_key_value(b), Some((&b, &1.0)));

        // Sanity check the length.
        assert_eq!(eigenvalue_centrality.len(), 2);

        // Two connections, degree centrality for A should increase.
        graph.insert(Edge::new(a, c));
        let eigenvalue_centrality = graph.eigenvalue_centrality();

        assert_eq!(
            eigenvalue_centrality.get_key_value(a),
            Some((&a, &1.2426406871192854))
        );
        assert_eq!(
            eigenvalue_centrality.get_key_value(b),
            Some((&b, &0.8786796564403571))
        );
        assert_eq!(
            eigenvalue_centrality.get_key_value(c),
            Some((&c, &0.8786796564403576))
        );

        // Sanity check the length.
        assert_eq!(eigenvalue_centrality.len(), 3);
    }

    #[test]
    fn fiedler() {
        let mut graph = Graph::new();

        let (a, b, c, d) = ("a", "b", "c", "d");

        // Disconnected graph.
        graph.insert(Edge::new(a, b));
        graph.insert(Edge::new(c, d));

        // Algebraic connectivity should be 0.
        let (algebraic_connectivity, fiedler_values_indexed) = graph.fiedler();
        assert_eq!(algebraic_connectivity, 0.0);
        assert_eq!(fiedler_values_indexed.get_key_value(a), Some((&a, &0.0)));
        assert_eq!(fiedler_values_indexed.get_key_value(b), Some((&b, &0.0)));
        assert_eq!(
            fiedler_values_indexed.get_key_value(c),
            Some((&c, &-0.7071067811865475))
        );
        assert_eq!(
            fiedler_values_indexed.get_key_value(d),
            Some((&d, &-0.7071067811865475))
        );

        // Connect the graph.
        graph.insert(Edge::new(b, c));

        let (algebraic_connectivity, fiedler_values_indexed) = graph.fiedler();
        assert_eq!(algebraic_connectivity, 0.5857864376269044);
        assert_eq!(
            fiedler_values_indexed.get_key_value(a),
            Some((&a, &0.6532814824381882))
        );
        assert_eq!(
            fiedler_values_indexed.get_key_value(b),
            Some((&b, &0.27059805007309845))
        );
        assert_eq!(
            fiedler_values_indexed.get_key_value(c),
            Some((&c, &-0.2705980500730985))
        );
        assert_eq!(
            fiedler_values_indexed.get_key_value(d),
            Some((&d, &-0.6532814824381881))
        );
    }

    //
    // Private
    //

    #[test]
    fn clear_cache_on_insert() {
        let mut graph = Graph::new();
        graph.insert(Edge::new("a", "b"));

        // The laplacian requires the computation of the index, the degree matrix and the adjacency
        // matrix.
        graph.laplacian_matrix();

        // Check the objects have been cached.
        assert!(graph.index.is_some());
        assert!(graph.adjacency_matrix.is_some());
        assert!(graph.degree_matrix.is_some());
        assert!(graph.laplacian_matrix.is_some());

        // Update the graph with an insert.
        graph.insert(Edge::new("a", "c"));

        // Check the cache has been cleared.
        assert!(graph.index.is_none());
        assert!(graph.adjacency_matrix.is_none());
        assert!(graph.degree_matrix.is_none());
        assert!(graph.laplacian_matrix.is_none());
    }

    #[test]
    fn clear_cache_on_subset_insert() {
        let mut graph = Graph::new();
        graph.insert(Edge::new("a", "b"));

        // The laplacian requires the computation of the index, the degree matrix and the adjacency
        // matrix.
        graph.laplacian_matrix();

        // Check the objects have been cached.
        assert!(graph.index.is_some());
        assert!(graph.adjacency_matrix.is_some());
        assert!(graph.degree_matrix.is_some());
        assert!(graph.laplacian_matrix.is_some());

        // Update the graph with a subset insert.
        graph.insert_subset("a", &["b", "d"]);

        // Check the cache has been cleared.
        assert!(graph.index.is_none());
        assert!(graph.adjacency_matrix.is_none());
        assert!(graph.degree_matrix.is_none());
        assert!(graph.laplacian_matrix.is_none());
    }

    #[test]
    fn clear_cache_on_subset_update() {
        let mut graph = Graph::new();
        graph.insert(Edge::new("a", "b"));

        // The laplacian requires the computation of the index, the degree matrix and the adjacency
        // matrix.
        graph.laplacian_matrix();

        // Check the objects have been cached.
        assert!(graph.index.is_some());
        assert!(graph.adjacency_matrix.is_some());
        assert!(graph.degree_matrix.is_some());
        assert!(graph.laplacian_matrix.is_some());

        // Update the graph with a subset update.
        graph.update_subset("a", &["b", "d"]);

        // Check the cache has been cleared.
        assert!(graph.index.is_none());
        assert!(graph.adjacency_matrix.is_none());
        assert!(graph.degree_matrix.is_none());
        assert!(graph.laplacian_matrix.is_none());
    }

    #[test]
    fn clear_cache_on_subset_update_w_only_removals() {
        let mut graph = Graph::new();
        graph.insert(Edge::new("a", "b"));
        graph.insert(Edge::new("a", "c"));

        // The laplacian requires the computation of the index, the degree matrix and the adjacency
        // matrix.
        graph.laplacian_matrix();

        // Check the objects have been cached.
        assert!(graph.index.is_some());
        assert!(graph.adjacency_matrix.is_some());
        assert!(graph.degree_matrix.is_some());
        assert!(graph.laplacian_matrix.is_some());

        // Update the graph with a subset update.
        graph.update_subset("a", &["b"]);

        // Check the cache has been cleared.
        assert!(graph.index.is_none());
        assert!(graph.adjacency_matrix.is_none());
        assert!(graph.degree_matrix.is_none());
        assert!(graph.laplacian_matrix.is_none());
    }

    #[test]
    fn clear_cache_on_remove() {
        let edge = Edge::new("a", "b");
        let mut graph = Graph::new();
        graph.insert(edge.clone());

        // The laplacian requires the computation of the index, the degree matrix and the adjacency
        // matrix.
        graph.laplacian_matrix();

        // Check the objects have been cached.
        assert!(graph.index.is_some());
        assert!(graph.adjacency_matrix.is_some());
        assert!(graph.degree_matrix.is_some());
        assert!(graph.laplacian_matrix.is_some());

        // Update the graph with remove.
        graph.remove(&edge);

        // Check the cache has been cleared.
        assert!(graph.index.is_none());
        assert!(graph.adjacency_matrix.is_none());
        assert!(graph.degree_matrix.is_none());
        assert!(graph.laplacian_matrix.is_none());
    }

    #[test]
    fn vertices_from_edges() {
        let mut graph = Graph::new();
        assert!(graph.vertices_from_edges().is_empty());

        let (a, b) = ("a", "b");
        graph.insert(Edge::new(a, b));

        let vertices = graph.vertices_from_edges();
        assert!(vertices.contains(a));
        assert!(vertices.contains(b));

        // Sanity check the length.
        assert_eq!(vertices.len(), 2);
    }

    #[test]
    fn generate_index() {
        let mut graph = Graph::new();

        // Check for an empty graph.
        graph.generate_index();
        assert!(graph.index.is_some());
        assert!(graph.index.as_ref().unwrap().is_empty());

        let (a, b) = ("a", "b");
        graph.insert(Edge::new(a, b));
        graph.generate_index();

        assert!(graph.index.is_some());

        assert_eq!(
            graph.index.as_ref().unwrap().get_key_value(a),
            Some((&a, &0))
        );

        assert_eq!(
            graph.index.as_ref().unwrap().get_key_value(b),
            Some((&b, &1))
        );

        assert_eq!(graph.index.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn closeness_randomish_graph() {
        let mut graph: Graph<usize> = Graph::new();
        // this graph reproduces the image at:
        // https://www.sotr.blog/articles/breadth-first-search
        graph.insert(Edge::new(0, 3));
        graph.insert(Edge::new(0, 5));
        graph.insert(Edge::new(5, 1));
        graph.insert(Edge::new(1, 2));
        graph.insert(Edge::new(2, 4));
        graph.insert(Edge::new(2, 6));
        graph.insert(Edge::new(1, 3));

        let between_map = graph.betweenness_centrality(2);
        let close_map = graph.closeness_centrality(2);
        let mut betweenness: [f64; 7] = [0.0; 7];
        let mut closeness: [f64; 7] = [0.0; 7];
        for i in 0..7 {
            betweenness[i] = *between_map.get(&i).unwrap();
            closeness[i] = *close_map.get(&i).unwrap();
        }

        let total_path_length = [28, 11, 13, 14, 19, 14, 19];
        let num_paths = [10, 7, 7, 7, 7, 7, 7];
        let total_num_paths: i32 = 52;
        let mut expected_closeness: [f64; 7] = [0.0; 7];
        let mut expected_betweenness: [f64; 7] = [0.0; 7];
        let betweenness_count = [1, 13, 11, 4, 0, 4, 0];
        for i in 0..7 {
            expected_closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
            expected_betweenness[i] = betweenness_count[i] as f64 / total_num_paths as f64;
        }

        assert_eq!(betweenness, expected_betweenness);
        assert_eq!(closeness, expected_closeness);
    }

    #[test]
    fn closeness_star_graph_a() {
        // 7-pointed star, 8 nodes
        // center is 0
        let mut graph: Graph<usize> = Graph::new();
        graph.insert(Edge::new(0, 1));
        graph.insert(Edge::new(0, 2));
        graph.insert(Edge::new(0, 3));
        graph.insert(Edge::new(0, 4));
        graph.insert(Edge::new(0, 5));
        graph.insert(Edge::new(0, 6));
        graph.insert(Edge::new(0, 7));

        let between_map = graph.betweenness_centrality(2);
        let close_map = graph.closeness_centrality(2);
        let mut betweenness: [f64; 8] = [0.0; 8];
        let mut closeness: [f64; 8] = [0.0; 8];
        for i in 0..8 {
            betweenness[i] = *between_map.get(&i).unwrap();
            closeness[i] = *close_map.get(&i).unwrap();
        }

        let total_path_length = [7, 13, 13, 13, 13, 13, 13, 13];
        let num_paths = [7, 7, 7, 7, 7, 7, 7, 7];
        let total_num_paths: i32 = 56;
        let mut expected_closeness: [f64; 8] = [0.0; 8];
        let mut expected_betweenness: [f64; 8] = [0.0; 8];
        let betweenness_count = [21, 0, 0, 0, 0, 0, 0, 0];
        for i in 0..8 {
            expected_closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
            expected_betweenness[i] = betweenness_count[i] as f64 / total_num_paths as f64;
        }
        assert_eq!(betweenness, expected_betweenness);
        assert_eq!(closeness, expected_closeness);
    }

    #[test]
    fn closeness_star_graph_b() {
        // 7-pointed star, 8 nodes
        // center is 7
        let mut graph: Graph<usize> = Graph::new();
        graph.insert(Edge::new(0, 7));
        graph.insert(Edge::new(1, 7));
        graph.insert(Edge::new(2, 7));
        graph.insert(Edge::new(3, 7));
        graph.insert(Edge::new(4, 7));
        graph.insert(Edge::new(5, 7));
        graph.insert(Edge::new(6, 7));

        let between_map = graph.betweenness_centrality(2);
        let close_map = graph.closeness_centrality(2);
        let mut betweenness: [f64; 8] = [0.0; 8];
        let mut closeness: [f64; 8] = [0.0; 8];
        for i in 0..8 {
            betweenness[i] = *between_map.get(&i).unwrap();
            closeness[i] = *close_map.get(&i).unwrap();
        }

        let total_path_length = [13, 13, 13, 13, 13, 13, 13, 7];
        let num_paths = [7, 7, 7, 7, 7, 7, 7, 7];
        let total_num_paths: i32 = 56;
        let mut expected_closeness: [f64; 8] = [0.0; 8];
        let mut expected_betweenness: [f64; 8] = [0.0; 8];
        let betweenness_count = [0, 0, 0, 0, 0, 0, 0, 21];
        for i in 0..8 {
            expected_closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
            expected_betweenness[i] = betweenness_count[i] as f64 / total_num_paths as f64;
        }
        assert_eq!(betweenness, expected_betweenness);
        assert_eq!(closeness, expected_closeness);
    }

    #[test]
    fn closeness_star_graph_c() {
        // 7-pointed star, 8 nodes
        // center is 7
        let mut graph: Graph<usize> = Graph::new();
        graph.insert(Edge::new(6, 3));
        graph.insert(Edge::new(4, 3));
        graph.insert(Edge::new(5, 3));
        graph.insert(Edge::new(1, 3));
        graph.insert(Edge::new(2, 3));
        graph.insert(Edge::new(7, 3));
        graph.insert(Edge::new(0, 3));

        let between_map = graph.betweenness_centrality(2);
        let close_map = graph.closeness_centrality(2);

        let mut betweenness: [f64; 8] = [0.0; 8];
        let mut closeness: [f64; 8] = [0.0; 8];
        for i in 0..8 {
            betweenness[i] = *between_map.get(&i).unwrap();
            closeness[i] = *close_map.get(&i).unwrap();
        }

        let total_path_length = [13, 13, 13, 7, 13, 13, 13, 13];
        let num_paths = [7, 7, 7, 7, 7, 7, 7, 7];
        let total_num_paths: i32 = 56;
        let mut expected_closeness: [f64; 8] = [0.0; 8];
        let mut expected_betweenness: [f64; 8] = [0.0; 8];
        let betweenness_count = [0, 0, 0, 21, 0, 0, 0, 0];
        for i in 0..8 {
            expected_closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
            expected_betweenness[i] = betweenness_count[i] as f64 / total_num_paths as f64;
        }
        assert_eq!(betweenness, expected_betweenness);
        assert_eq!(closeness, expected_closeness);
    }

    #[test]
    fn closeness_disconnected_graph() {
        // 9 vertices
        // 4 verts, 0-3: square, all points connected
        // 5 verts, 4-8: star, with v4 in the center
        let mut graph: Graph<usize> = Graph::new();
        graph.insert(Edge::new(0, 1));
        graph.insert(Edge::new(0, 2));
        graph.insert(Edge::new(0, 3));
        graph.insert(Edge::new(1, 2));
        graph.insert(Edge::new(1, 3));
        graph.insert(Edge::new(2, 3));
        graph.insert(Edge::new(4, 5));
        graph.insert(Edge::new(4, 6));
        graph.insert(Edge::new(4, 7));
        graph.insert(Edge::new(4, 8));

        let between_map = graph.betweenness_centrality(2);
        let close_map = graph.closeness_centrality(2);
        let mut betweenness: [f64; 9] = [0.0; 9];
        let mut closeness: [f64; 9] = [0.0; 9];
        for i in 0..9 {
            betweenness[i] = *between_map.get(&i).unwrap();
            closeness[i] = *close_map.get(&i).unwrap();
        }

        let total_path_length = [3, 3, 3, 3, 4, 7, 7, 7, 7];
        let num_paths = [3, 3, 3, 3, 4, 4, 4, 4, 4];
        let total_num_paths: i32 = 32;
        let mut expected_closeness: [f64; 9] = [0.0; 9];
        let mut expected_betweenness: [f64; 9] = [0.0; 9];
        let betweenness_count = [0, 0, 0, 0, 6, 0, 0, 0, 0];
        for i in 0..9 {
            expected_closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
            expected_betweenness[i] = betweenness_count[i] as f64 / total_num_paths as f64;
        }
        assert_eq!(betweenness, expected_betweenness);
        assert_eq!(closeness, expected_closeness);
    }

    // Helper function to create a sample from a json file.
    // The file will begin like this:
    //   {"indices":[[2630,3217,1608,1035,...
    // and end like this:
    //   ...2316,1068,1238,704,2013]]}
    pub fn load_sample(filepath: &str) -> Sample {
        let jstring = fs::read_to_string(filepath).unwrap();
        let sample: Sample = serde_json::from_str(&jstring).unwrap();
        sample
    }

    #[test]
    fn loaded_sample_graph() {
        let sample = load_sample("testdata/sample.json");

        // graph 1 uses integers as node value
        let mut graph1 = Graph::new();
        let mut n = 0;
        for node in &sample.indices {
            for connection in node {
                if *connection > n {
                    graph1.insert(Edge::new(n, *connection));
                }
            }
            n += 1;
        }
        // _ = graph1.betweenness_centrality(1);
        // graph1.clear_cache();
        // _ = graph1.betweenness_centrality(2);
        // graph1.clear_cache();
        _ = graph1.betweenness_centrality(3);
        graph1.clear_cache();
        _ = graph1.betweenness_centrality(4);
        graph1.clear_cache();
        _ = graph1.betweenness_centrality(5);
        graph1.clear_cache();
        let betweenness_centrality1 = graph1.betweenness_centrality(6);

        // let betweenness_centrality1 = graph1.betweenness_centrality(3);
        let closeness_centrality1 = graph1.closeness_centrality(3);

        // graph2 uses ip address as node value
        let mut graph2: Graph<&str> = Graph::new();
        let mut n = 0;
        for node in &sample.indices {
            for connection in node {
                if *connection > n {
                    graph2.insert(Edge::new(
                        &sample.node_ips[n],
                        &sample.node_ips[*connection],
                    ));
                }
            }
            n += 1;
        }

        let betweenness_centrality2 = graph2.betweenness_centrality(5);
        let closeness_centrality2 = graph2.closeness_centrality(5);
        let b1 = betweenness_centrality1.get(&0).unwrap();
        let b2 = betweenness_centrality2.get("65.21.141.242").unwrap();
        let c1 = closeness_centrality1.get(&0).unwrap();
        let c2 = closeness_centrality2.get("65.21.141.242").unwrap();
        assert_eq!(b1, b2);
        assert_eq!(c1, c2);

        // Index 1837 has betweenness 9.576638518159478e-8
        // we'll confirm it's between 0.00000009 and 0.00000010
        let b1 = betweenness_centrality1.get(&1837).unwrap();
        let b2 = betweenness_centrality2.get("85.15.179.171").unwrap();
        let c1 = closeness_centrality1.get(&1837).unwrap();
        let c2 = closeness_centrality2.get("85.15.179.171").unwrap();
        assert_eq!(b1, b2);
        assert_eq!(c1, c2);
        assert!(*b1 > 0.00000009);
        assert!(*b1 < 0.00000010);

        // these should not be equal
        let b1 = betweenness_centrality1.get(&1836).unwrap();
        let b2 = betweenness_centrality2.get("85.15.179.171").unwrap();
        assert_ne!(b1, b2);
    }

    #[test]
    fn betweenness_line_topology() {
        let (a, b, c, d) = ("a", "b", "c", "d");
        let mut graph = graph!([a, b, c, d]);

        let betweenness_centrality = graph.betweenness_centrality(2);

        assert_eq!(betweenness_centrality.get_key_value(a), Some((&a, &0.0)));
        assert_eq!(
            betweenness_centrality.get_key_value(b),
            Some((&b, &(2.0 / 12.0)))
        );
        assert_eq!(
            betweenness_centrality.get_key_value(c),
            Some((&c, &(2.0 / 12.0)))
        );
        assert_eq!(betweenness_centrality.get_key_value(d), Some((&d, &0.0)));
    }

    #[test]
    fn betweenness_star_topology() {
        let (a, b, c, d, e) = ("a", "b", "c", "d", "e");
        let mut graph = graph!([a, b, c], [e, b, d]);

        let betweenness_centrality = graph.betweenness_centrality(2);

        assert_eq!(betweenness_centrality.get_key_value(a), Some((&a, &0.0)));
        assert_eq!(
            betweenness_centrality.get_key_value(b),
            Some((&b, &(6.0 / 20.0)))
        );
        assert_eq!(betweenness_centrality.get_key_value(c), Some((&c, &0.0)));
        assert_eq!(betweenness_centrality.get_key_value(d), Some((&d, &0.0)));
        assert_eq!(betweenness_centrality.get_key_value(e), Some((&e, &0.0)));
    }
}
