# Graph based clustering algorithm

- [dayyass](https://github.com/dayyass)/**[graph-based-clustering](https://github.com/dayyass/graph-based-clustering)** 

- wikipedia [HCS clustering algorithm](https://en.wikipedia.org/wiki/HCS_clustering_algorithm) 

## Example

Q: i have some points, and a given vector, write a java program to cluster points to group accourding to if the vetor of line segment connecting the two points is nearly perpendicular to the give vector

A: Certainly! Here’s a complete **Java program** to read some points and a reference vector, and group/cluster the points according to whether the line segment joining two points forms a vector that is **nearly perpendicular** (within a threshold angle) to the given vector.

For **clustering**, we'll treat each pair satisfying the perpendicular condition as an **edge** in a graph. Clusters are then the **connected components** of this graph.

---

### Full Example

```java
import java.awt.geom.Point2D;
import java.util.*;

public class PerpendicularPointCluster {

    // Params
    static double DEGREE_THRESHOLD = 10.0;              // Threshold in degrees
    static double COSINE_THRESHOLD = Math.abs(Math.cos(Math.toRadians(90 - DEGREE_THRESHOLD)));

    // Returns true if vector ab is nearly perpendicular to vector v
    static boolean isNearlyPerpendicular(Point2D.Double a, Point2D.Double b, double[] v) {
        double dx = b.x - a.x;
        double dy = b.y - a.y;
        double[] seg = {dx, dy};
        double segMag = Math.hypot(dx, dy);
        double vMag = Math.hypot(v[0], v[1]);
        if (segMag == 0 || vMag == 0) return false; // Degenerate case
        // Cosine of angle between vectors:
        double dot = dx * v[0] + dy * v[1];
        double cosTheta = dot / (segMag * vMag);
        // Check if angle between segment and v is near 90 degrees
        return Math.abs(cosTheta) < COSINE_THRESHOLD;
    }

    // Finds clusters (connected components) of points based on perpendicular edges
    static List<List<Integer>> findClusters(List<Point2D.Double> points, double[] v) {
        int n = points.size();
        // Build adjacency list where there is an edge if nearly perpendicular
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; ++i) adj.add(new ArrayList<>());
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (isNearlyPerpendicular(points.get(i), points.get(j), v)) {
                    adj.get(i).add(j);
                    adj.get(j).add(i);
                }
            }
        }
        // BFS/DFS for connected components
        boolean[] visited = new boolean[n];
        List<List<Integer>> clusters = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                List<Integer> cluster = new ArrayList<>();
                Deque<Integer> stack = new ArrayDeque<>();
                stack.push(i);
                while (!stack.isEmpty()) {
                    int u = stack.pop();
                    if (!visited[u]) {
                        visited[u] = true;
                        cluster.add(u);
                        for (int vtx : adj.get(u)) {
                            if (!visited[vtx]) stack.push(vtx);
                        }
                    }
                }
                clusters.add(cluster);
            }
        }
        return clusters;
    }

    public static void main(String[] args) {
        // Example points; you can load these from user/file as needed
        List<Point2D.Double> points = Arrays.asList(
                new Point2D.Double(0, 0),
                new Point2D.Double(1, 1),
                new Point2D.Double(2, 0),
                new Point2D.Double(1, -1),
                new Point2D.Double(0, 2),
                new Point2D.Double(3, 0),
                new Point2D.Double(0, 3)
        );

        // Reference vector: e.g., along x-axis
        double[] referenceVector = {1, 0};

        List<List<Integer>> clusters = findClusters(points, referenceVector);

        // Print clusters
        int num = 1;
        for (List<Integer> cluster : clusters) {
            System.out.print("Cluster " + num + ": ");
            for (int idx : cluster) {
                System.out.print("(" + points.get(idx).x + ", " + points.get(idx).y + ") ");
            }
            System.out.println();
            num++;
        }
    }
}
```

---

### How it works:

- For each pair of points, it checks if the vector connecting them is nearly perpendicular to the reference vector.
- It creates edges between such pairs.
- It finds all connected components (clusters) using DFS.

---

### Customizing:

- **Threshold Angle**: Change `DEGREE_THRESHOLD` for stricter/looser perpendicularity.
- **Input**: Read points and vector from input or file as needed.
- **Output**: Prints each cluster with member point coordinates.

---



### Q: strictly speaking, what does the cluster algorithm you just mentioned belong to

Strictly speaking, the clustering algorithm described in the previous answer is an example of **graph-based clustering**. More specifically, it is called **connected component clustering** (or single-link clustering in graph theory).

#### Why?

- Each point is a **node**.
- There is an **edge** between two points if the segment connecting them satisfies a certain condition (in your case, "nearly perpendicular to a reference vector").
- The clustering consists of finding **connected components** in this undirected graph: each cluster is a set of points that are all reachable from each other via such edges.

#### In summary:

> **The algorithm is "graph-based connected component clustering," also called "single-linkage clustering" (when under a similarity or adjacency rule).**

#### Related concepts:

- _Hierarchical clustering_: Single-linkage can be viewed as a hierarchical clustering, but in this context, it is just grouping by direct connection.
- _Agglomerative clustering_: Single-link in graph theory is equivalent to agglomerative clustering with the minimum linkage criterion.
- _Community detection in graphs_: Connected components are the simplest "communities."

---

**References:**

- [Single-linkage clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering#Single-linkage)
- [Connected component](https://en.wikipedia.org/wiki/Connected_component_(graph_theory))

---

#### Summary Table

| Criterion                  | Category               |
| -------------------------- | ---------------------- |
| Based on connectivity      | Graph-based clustering |
| Detecting connected groups | Connected components   |
| Based on pairwise criteria | Single-linkage method  |

---

If you need a more formal citation or reference to clustering taxonomy, just ask!


