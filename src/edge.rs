use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

/// A pair of vertices representing a graph edge. Edges don't have a direction, despite the
/// `source`-`target` nomenclature used.
#[derive(Clone, Debug, Eq)]
pub struct Edge<T> {
    source: T,
    target: T,
}

// Core implementation

impl<T> Edge<T> {
    /// Creates a new edge from two vertices.
    pub fn new(source: T, target: T) -> Self {
        Self { source, target }
    }

    /// Returns the first vertice forming the edge.
    pub fn source(&self) -> &T {
        &self.source
    }

    /// Returns the second vertice forming the edge.
    pub fn target(&self) -> &T {
        &self.target
    }
}

// Trait implementations

impl<T: PartialEq> PartialEq for Edge<T> {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (&self.source, &self.target);
        let (c, d) = (&other.source, &other.target);

        a == d && b == c || a == c && b == d
    }
}

impl<T: Hash + Ord> Hash for Edge<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let (a, b) = (&self.source, &self.target);

        // This ensures the hash is the same for (a, b) as it is for (b, a).
        match a.cmp(b) {
            Ordering::Greater => {
                b.hash(state);
                a.hash(state);
            }
            _ => {
                a.hash(state);
                b.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Core implementation

    #[test]
    fn new() {
        let (source, target) = ("a", "b");

        assert_eq!(Edge::new(source, target), Edge { source, target })
    }

    #[test]
    fn source() {
        let (a, b) = ("a", "b");
        let edge = Edge::new(a, b);

        assert_eq!(edge.source(), &a);
    }

    #[test]
    fn target() {
        let (a, b) = ("a", "b");
        let edge = Edge::new(a, b);

        assert_eq!(edge.target(), &b);
    }

    // Trait implementations

    #[test]
    fn partial_eq() {
        let (a, b) = ("a", "b");

        assert_eq!(Edge::new(a, b), Edge::new(a, b));
        assert_eq!(Edge::new(a, b), Edge::new(b, a));
    }

    #[test]
    fn hash() {
        use std::collections::hash_map::DefaultHasher;

        let (a, b) = ("a", "b");

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();

        let k1 = Edge::new(a, b);
        let k2 = Edge::new(b, a);

        k1.hash(&mut h1);
        k2.hash(&mut h2);

        // Verify k1 == k2 => hash(k1) == hash(k2).
        assert_eq!(h1.finish(), h2.finish());
    }
}