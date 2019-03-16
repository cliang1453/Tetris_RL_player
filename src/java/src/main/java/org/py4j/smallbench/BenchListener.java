package org.py4j.smallbench;

public interface BenchListener {

	// void notify(int next_piece, boolean is_end);
	void notify(int next_piece, int[][] field, boolean is_end);
}
