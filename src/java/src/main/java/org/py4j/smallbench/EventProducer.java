package org.py4j.smallbench;

import java.util.concurrent.BlockingDeque;
import java.util.concurrent.BlockingQueue;

public class EventProducer implements Runnable {

	private PlayerSkeleton p;
	
	public EventProducer(PlayerSkeleton player) {
		p = player;
	}

	public static void startGame(PlayerSkeleton player) {

		System.out.println("EventProducer:startGame...");

		EventProducer producer = new EventProducer(player);

		Thread t = new Thread(producer);
		t.start();

	}

	public void run() {
		
		try {

			System.out.println("EventProducer:run...");
			int[] a = p.getAction();
			State s = p.getState();
			System.out.println("EventProducer:a" + a[0] + " " + a[1]);
			
			while (!s.hasLost()){
				s.makeMove(a[0], a[1]);
				s.draw();
				s.drawNext(0,0);
				p.notifyAllListeners();
				Thread.currentThread().sleep(10);
			}
			System.out.println(s.getRowsCleared());
			

		} 
		catch (Exception e) {
				e.printStackTrace();
		}
	}
}
