package org.py4j.smallbench;

public interface BenchListener {

	Object notify(int next_piece, int[][] field, boolean is_end);
}
