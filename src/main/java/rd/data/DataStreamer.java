package rd.data;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.stream.Stream;

import org.jblas.FloatMatrix;

/**
 * Data Streamer - stream out inputs one at a time, also possible to get random
 * entries Each new input vector is stored as a column vector. Thus each column
 * is a new input
 * 
 * @author azahar
 *
 */
public class DataStreamer implements Iterable<FloatMatrix> {

	private final Map<FloatMatrix, FloatMatrix> streamData = new HashMap<>();

	private final int inDataWidth, outDataWidth;

	/**
	 * 
	 * @param inDataWidth
	 *            - input data width
	 * @param outDataWidth
	 *            - output data width
	 */
	public DataStreamer(int inDataWidth, int outDataWidth) {
		this.inDataWidth = inDataWidth;
		this.outDataWidth = outDataWidth;
	}

	/**
	 * 
	 * @param data
	 *            - Map of input and output float matrix (rows)
	 */
	public DataStreamer(Map<FloatMatrix, FloatMatrix> data) {
		FloatMatrix key = data.keySet().iterator().next();
		this.inDataWidth = key.getRows();
		this.outDataWidth = data.get(key).getRows();
		streamData.putAll(data);
	}

	/**
	 * Add input output pair
	 * 
	 * @param data
	 * @param output
	 */
	public void add(float[] data, float... output) {
		if (data.length == inDataWidth) {
			FloatMatrix inFm = new FloatMatrix(inDataWidth, 1);
			FloatMatrix outFm = new FloatMatrix(outDataWidth, 1);
			int i = 0;
			for (float item : data) {
				inFm.put(i++, 0, item);
			}
			i = 0;
			for (float item : output) {
				outFm.put(i++, item);
			}
			synchronized (this) {
				streamData.put(inFm, outFm);
			}
		} else {
			System.err.println("Error data width does not match");
		}
	}

	/**
	 * Add input output pair
	 * 
	 * @param data
	 * @param output
	 */
	public void add(int[] data, int... output) {
		if (data.length == inDataWidth) {
			FloatMatrix inFm = new FloatMatrix(inDataWidth, 1);
			FloatMatrix outFm = new FloatMatrix(outDataWidth, 1);
			int i = 0;
			for (int item : data) {
				inFm.put(i++, 0, (float) item);
			}
			i = 0;
			for (int item : output) {
				outFm.put(i++, (float) item);
			}
			synchronized (this) {
				streamData.put(inFm, outFm);
			}
		} else {
			System.err.println("Error data width does not match");
		}
	}

	/**
	 * Split into two Data Streamers - for splitting up data into test and
	 * training sets
	 * 
	 * @param ratio
	 *            -the ratio of split, value 0.0 > and < 1.0 - if split > 0.5
	 *            then DataStreamer in 0 position is bigger
	 * @return split data streamer
	 */
	public DataStreamer[] split(float ratio) {

		Map<FloatMatrix, FloatMatrix> _1 = new HashMap<>(), _2 = new HashMap<>();
		for (FloatMatrix item : this) {
			if (Math.random() <= ratio) {
				_1.put(item, this.getOutput(item));
			} else {
				_2.put(item, this.getOutput(item));
			}
		}

		return new DataStreamer[] { new DataStreamer(_1), new DataStreamer(_2) };
	}

	/**
	 * Randomly return an input
	 * 
	 * @return randomly selected input - use getOutput to obtain corresponding
	 *         output
	 */
	public FloatMatrix getRandom() {
		int index = (int) (Math.random() * streamData.keySet().size());
		FloatMatrix fm = null;
		int i = 0;
		for (FloatMatrix key : streamData.keySet()) {
			if (i >= index) {
				return key;
			}
			i++;
			fm = key;
		}

		return fm;
	}

	/**
	 * Get number of inputs (unique)
	 * 
	 * @return
	 */
	public int getNumberOfUniqueInputs() {
		return streamData.keySet().size();
	}

	/**
	 * Data Streamer to CSV Stream
	 * 
	 * @return CSV Stream
	 */
	public Stream<String> toCsvStream() {
		return streamData.keySet().stream()
				.map(item -> item.toString().replaceAll(";", ",").replaceAll("\\[", "").replaceAll("\\]", "") + ","
						+ getOutput(item).toString().replaceAll("\\[", "").replaceAll("\\]", ""));

	}

	@Override
	/**
	 * Iterator - returns iterator for data
	 * 
	 * @return data streamer
	 */
	public Iterator<FloatMatrix> iterator() {
		return streamData.keySet().iterator();
	}

	/**
	 * Get output from input
	 * 
	 * @param input
	 *            - the input for which we need to get the output
	 * @return get output
	 */
	public FloatMatrix getOutput(FloatMatrix input) {
		return streamData.get(input);
	}

	/**
	 * Get Input as Array
	 * 
	 * @return
	 */
	public FloatMatrix[] getInputAsArray() {
		return (FloatMatrix[]) streamData.keySet().toArray();
	}

	public int size() {
		return streamData.keySet().size();
	}
}
