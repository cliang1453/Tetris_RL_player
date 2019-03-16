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
			State s = p.getState();
			// System.out.println("EventProducer:a" + a[0] + " " + a[1]);
			
			int game_count = 0;
			while (game_count<10){
				int[] a = p.getAction();
				s.makeMove(a[0], a[1]);
				s.draw();
				s.drawNext(0,0);
				if (s.hasLost()) {
					game_count += 1;
					p.notifyAllListeners();
					s = p.getState();
				}
				else {
					p.notifyAllListeners();
				}
				
				Thread.currentThread().sleep(10);

				System.out.println("game count"+game_count);
			}
			System.out.println(s.getRowsCleared());
			

		} 
		catch (Exception e) {
				e.printStackTrace();
		}
	}
}
