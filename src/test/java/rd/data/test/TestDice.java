package rd.data.test;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class TestDice {

	private static final int QTY = 4;
	private static Map<String, Integer> counts = new HashMap<>();

	public static void main(String... args) {
		int count = 0;
		double maxClass = Math.pow(6, QTY);
		Dice[] cache = new Dice[QTY];
		for (int i = 0; i < QTY; i++) {
			
			if(i<QTY)
			{
				cache[i] = (Dice) new BiasedDice(new int[]{1,6});
			}
			else
			{
				cache[i] = (Dice) new FairDice();
			}
		}
		for (int i = 0; i < 1000000; i++) {
			String str = "";
			for (int j = 0; j < QTY; j++) {
				str += cache[j].roll()+"|";
			}
			count++;
			update(counts, str);

		}
		

		
		for(String key : counts.keySet())
		{
			System.out.println(key+", "+counts.get(key));
		}
		
		System.out.println(">> Coverage (%): "+(counts.keySet().size()*100/maxClass));
	}

	private static void update(Map<String, Integer> map, String key) {
		Integer val = map.get(key);
		if (val == null) {
			map.put(key, 1);
		} else {
			map.put(key, val + 1);
		}
	}

}

interface Dice {
	int roll();
}

class FairDice implements Dice {
	private Random rnd;

	public FairDice() {
		rnd = new Random();
	}

	@Override
	public int roll() {
		return 1 + rnd.nextInt(6);
	}
}

class BiasedDice implements Dice {

	private Random rnd;
	private int excl[];

	public BiasedDice(int...excl) {
		this.excl = excl;
		rnd = new Random();
	}

	@Override
	public int roll() {
	
		int val =  1 + rnd.nextInt(6);
		if(excl.length>0 && rnd.nextDouble()<0.3)
		{
			return excl[rnd.nextInt(excl.length)];
		}
		
		return val;
		
	}

}