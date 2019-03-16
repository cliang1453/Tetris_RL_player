package org.py4j.smallbench;

import java.util.concurrent.BlockingDeque;
import java.util.concurrent.BlockingQueue;

public class EventProducer implements Runnable {

	private SmallBenchApplication application;

	private BlockingQueue<Object> queue;

	public EventProducer(SmallBenchApplication application) {
		this.application = application;
		this.queue = application.getQueue();
	}

	public static void startProducer(SmallBenchApplication application) {

		System.out.println("EventProducer:startProducer...");

		EventProducer producer = new EventProducer(application);

		Thread t = new Thread(producer);
		t.start();

	}

	public void run() {
		System.out.println("EventProducer:run...");
		for (int i = 0; i < 5; i++) {
			try {
				System.out.println("	EventProducer:run"+i);
				int message = this.application.getMsg();
				System.out.println("	EventProducer:message"+message);
				queue.add("Testing from " + Thread.currentThread().getName());
				application.notifyAllListeners();
				Thread.currentThread().sleep(10);

			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
}
