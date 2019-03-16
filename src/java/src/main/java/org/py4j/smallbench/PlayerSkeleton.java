package org.py4j.smallbench;

import py4j.GatewayServer;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import java.util.*;
import java.io.*;
import py4j.GatewayServer;
// export CLASSPATH=/usr/local/share/py4j/py4j0.10.8.1.jar:.
// export CLASSPATH=/home/chen/.local/share/py4j/py4j0.10.8.1.jar:.

public class PlayerSkeleton {
	//implement this function to have a working system

	private int slot; 
	private int orient;
	private static State s;
	private static TFrame t; 
	private List<BenchListener> listeners = new ArrayList<BenchListener>();
	

	// public int pickMove(State s, int[][] legalMoves) {
	// 	Random generator = new Random();
	// 	int randomIndex = generator.nextInt(legalMoves.length);
	// 	return randomIndex;
	// }

	public int[] getAction() {
		int[] action = {orient, slot};
		return action;
	}

	public State getState(){
		return s;
	}

	public void registerBenchListener(BenchListener listener) {
		listeners.add(listener);
	}

	public void startGames(int count) {
		System.out.println("PlayerSkeleton:startGame...");
		for (int i = 0; i < count; i++) {
			System.out.println("In startGame"+i);
			EventProducer.startGame(this);
		}
	}

	public void takeAction(int orient, int slot) {
		System.out.println("PlayerSkeleton:takeAction" + slot + " " + orient);
		this.slot = slot; 
		this.orient = orient;
	}

	

	public void notifyAllListeners() {
		System.out.println("SmallBench:notifyAllListeners");
		
		try {
			System.out.println("0");
			int next_piece = s.getNextPiece();
			System.out.println("1");
			int[][] field = new int[s.COLS][s.ROWS];
			field = s.getField();
			System.out.println("2");
			boolean is_end = s.hasLost();
			System.out.println("3");
			for (BenchListener listener : listeners) {
				System.out.println("in for loop");
				listener.notify(next_piece, field, is_end);
				System.out.println("finish one loop");
			}
			System.out.println("completed");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}



	public static void main(String[] args) {
		PlayerSkeleton p = new PlayerSkeleton();
		GatewayServer server = new GatewayServer(p);
		server.start();
		
		s = new State();
		t = new TFrame(s);
	}
}