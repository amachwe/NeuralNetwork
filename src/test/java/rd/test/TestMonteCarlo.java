package rd.test;

/**
 * Monte Carlo - value of Pi - Grain of Rice experiment
 * https://en.wikipedia.org/wiki/Monte_Carlo_method
 * 
 * @author azahar
 *
 */
public class TestMonteCarlo {

	private static final float MAX_TRIES = 1000000, MAX_REPS = 100;

	public static void main(String... args) {
		for (int reps = 0; reps < MAX_REPS; reps++) {
			double dia = 10, xc = 5, yc = 5;
			float countC = 0;

			for (int tries = 0; tries < MAX_TRIES; tries++) {
				double x = gRand(dia, 0);
				double y = gRand(dia, 0);
				double rad = Math.sqrt(Math.pow((x - xc), 2) + Math.pow((y - yc), 2));

				if (rad <= (dia / 2f)) {
					countC++;
				}

			}

			float result = 4 * countC / MAX_TRIES;
			System.out.println(Math.PI + ", " + result + ", " + (Math.PI - result));
		}
	}

	private static double gRand(double max, double min) {
		return ((max - min) * Math.random()) + min;
	}
}
