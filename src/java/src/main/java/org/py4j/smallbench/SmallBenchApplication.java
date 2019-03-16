package org.py4j.smallbench;

import py4j.GatewayServer;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class SmallBenchApplication {

	private BlockingQueue<Object> queue = new ArrayBlockingQueue<Object>(1000);

	private List<BenchListener> listeners = new ArrayList<BenchListener>();

	private int message = 0;

	public BlockingQueue<Object> getQueue() {
		return queue;
	}

	public int getMsg() {
		System.out.println("	SmallBench:getMsg"+message);
		return message;
	}

	public void registerBenchListener(BenchListener listener) {
		listeners.add(listener);
	}

	public void notifyAllListeners() {
		System.out.println("	SmallBench:notifyAllListeners");
		try {
			Object object = queue.take();
			for (BenchListener listener : listeners) {
				listener.notify(object);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void startProducers(int count) {
		System.out.println("SmallBench:startProducers...");
		for (int i = 0; i < count; i++) {
			System.out.println("In startProducers"+i);
			EventProducer.startProducer(this);
		}
	}

	public void takeMsg(int msg) {
		System.out.println("SmallBench:takeMsg"+msg);
		message = msg;
	}

	public static void main(String[] args) {
		SmallBenchApplication application = new SmallBenchApplication();

		GatewayServer server = new GatewayServer(application);
		server.start();
	}

}
