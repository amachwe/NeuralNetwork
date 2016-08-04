package rd.data;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import au.com.bytecode.opencsv.CSVReader;
/**
 * Flatten a class column using one hot encoding
 * Example: Single class column with say 3 different values
 * gets flattened to a length 3 vector using one-hot encoding
 * 
 * @author azahar
 *
 */
public class CSVToFlatClassDataStreamer {
	private final CSVReader reader;

	/**
	 * 
	 * @param filename = of the CSV file
	 * @throws FileNotFoundException
	 */
	public CSVToFlatClassDataStreamer(String filename) throws FileNotFoundException {
		reader = new CSVReader(new FileReader(filename));

	}

	/**
	 * Get the flat representation - input single output column output one hot encoded 
	 * @param inputSize - input size (number of elements)
	 * @param outputSize - output size required (number of elements = total different class labels)
	 * @param handler - handler to flatten the column representation
	 * @return flattend data stremaer
	 * @throws IOException
	 */
	public DataStreamer getFlatClassDataStreamer(int inputSize, int outputSize, ClassHandler handler) throws IOException {
		DataStreamer ds = new DataStreamer(inputSize, outputSize);
		String[] data;
		while ((data = reader.readNext()) != null) {

			float input[] = new float[inputSize];
			float output[] = null;
			for (int i = 0; i < data.length; i++) {
				if (i < inputSize) {
					input[i] = Float.parseFloat(data[i].trim());
				} else {
					output = handler.getFlatClass(data[i].trim(), outputSize);
				}
			}

			ds.add(input, output);
		}
		return ds;
	}
}
