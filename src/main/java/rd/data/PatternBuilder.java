package rd.data;

import java.util.Random;

import org.jblas.FloatMatrix;

public class PatternBuilder {

	private final Random rnd;

	public PatternBuilder(Random rnd) {

		this.rnd = rnd;
	}

	public PatternBuilder() {

		this.rnd = new Random(123);
	}

	public DataStreamer getDataSet(int unitLength, int numberOfUnits, int count, float noise) {
		int length = unitLength * numberOfUnits;
		DataStreamer ds = new DataStreamer(length, 1);

		for (int i = 0; i < count; i++) {
			float[] fm = new float[length];
			int index = rnd.nextInt(numberOfUnits);
			int vecIndex = 0;
			for (int j = 0; j < numberOfUnits; j++) {
				for (int k = 0; k < unitLength; k++) {
					if (j == index) {
						fm[vecIndex] = decide(1.0f - noise);
					} else {
						fm[vecIndex] = 0;
					}
					vecIndex++;
				}
			}

			ds.add(fm, rnd.nextFloat());
		}

		return ds;
	}

	public static final float score(FloatMatrix a, FloatMatrix b) {
		float score = 0;
		for (int i = 0; i < a.length; i++) {
			if (a.get(i) == b.get(i)) {
				score++;
			}
		}

		return score / b.length;
	}

	public static final float score(FloatMatrix a, FloatMatrix b, float threshold) {
		float score = 0;
		for (int i = 0; i < a.length; i++) {
			if (Math.abs(a.get(i) - b.get(i)) <= threshold) {
				score++;
			}
		}

		return score / b.length;
	}

	public static final float matchScore(FloatMatrix a, FloatMatrix b) {
	
		float maxA = 0, maxB = 0, tmpA = 0, tmpB = 0;
		int indexA = 0, indexB = 0;
		for (int i = 0; i < a.length; i++) {
			tmpA = a.get(i);
			tmpB = b.get(i);
			if (tmpA > maxA) {
				maxA = tmpA;
				indexA=i;
			}
			if (tmpB > maxB) {
				maxB = tmpB;
				indexB=i;
			}
		}

		return indexA == indexB ? 1:0;
	}

	public final float decide(float prob) {
		if (rnd.nextDouble() < prob) {
			return 1.0f;
		} else {
			return 0.0f;
		}
	}

	public static void main(String... args) {
		PatternBuilder pb = new PatternBuilder();

		pb.getDataSet(4, 3, 1000, 0.05f).forEach(f -> System.out.println(f));
	}
}
