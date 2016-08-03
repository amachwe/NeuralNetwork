package rd.data;

/**
 * Confusion Matrix Predicted vs Actual results
 * 
 * @author azahar
 *
 */
public class ConfusionMatrix {

	private final int matrix[][] = new int[2][2];
	private int count = 0;

	public ConfusionMatrix() {
		matrix[0][0] = 0;
		matrix[0][1] = 0;
		matrix[1][0] = 0;
		matrix[1][1] = 0;
	}

	/**
	 * Increment True Positive Count
	 */
	public void incTP() {
		matrix[0][0]++;
		count++;
	}

	/**
	 * Get True Positive Count
	 * 
	 * @return - true positive count
	 */
	public int getTP() {
		return matrix[0][0];

	}

	/**
	 * Increment False Negative Count
	 */
	public void incFN() {
		matrix[0][1]++;
		count++;
	}

	/**
	 * Get False Negative Count
	 * 
	 * @return - false negative count
	 */
	public int getFN() {
		return matrix[0][1];

	}

	/**
	 * Increment True Negative Count
	 */
	public void incTN() {
		matrix[1][1]++;
		count++;
	}

	/**
	 * Get True Negative Count
	 * 
	 * @return - true negative count
	 */
	public int getTN() {
		return matrix[1][1];

	}

	/**
	 * Increment False Positive Count
	 */
	public void incFP() {
		matrix[1][0]++;
		count++;
	}

	/**
	 * Get False Positive Count
	 * 
	 * @return - false positive count
	 */
	public int getFP() {
		return matrix[1][0];

	}

	/**
	 * Total Cases Count
	 * 
	 * @return - total cases count
	 */
	public int getCount() {
		return count;
	}

	/**
	 * Accuracy = TP + TN / TOTAL COUNT
	 * 
	 * @return
	 */
	public float getAccuracy() {
		return count == 0 ? 0 : (float) (matrix[0][0] + matrix[1][1]) / (float) count;
	}

	/**
	 * Precision = TP / (TP + FP)
	 * 
	 * @return
	 */
	public float getPrecision() {
		return (float) (matrix[0][0] + matrix[0][1]) == 0 ? 0
				: (float) (matrix[0][0]) / (float) (matrix[0][0] + matrix[0][1]);
	}

	/**
	 * Recall = TP / (TP + FN)
	 * 
	 * @return
	 */
	public float getRecall() {
		return (float) (matrix[0][0] + matrix[1][0]) == 0 ? 0
				: (float) (matrix[0][0]) / (float) (matrix[0][0] + matrix[1][0]);
	}

	/**
	 * String representation of Confusion Matrix
	 * 
	 */
	@Override
	public String toString() {
		return "Accuracy: " + this.getAccuracy() + ",\tRecall: " + this.getRecall() + ",\tPrecision: "
				+ this.getPrecision();
	}

}
