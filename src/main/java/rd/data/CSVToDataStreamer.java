package rd.data;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import au.com.bytecode.opencsv.CSVReader;
/**
 * CSV To Data Streamer
 * @author azahar
 *
 */
public class CSVToDataStreamer {
	private final CSVReader reader;

	/**
	 * 
	 * @param filename - the CSV file
	 * @throws FileNotFoundException
	 */
	public CSVToDataStreamer(String filename) throws FileNotFoundException {
		reader = new CSVReader(new FileReader(filename));

	}

	/**
	 * Direct mapping of input and output
	 * @param inputSize - number of elements in Input
	 * @param outputSize - number of elements in Output
	 * @param handler - conversion handler (convert string based class to integer)
	 * @return Data Stream
	 * @throws IOException
	 */
	public DataStreamer getDataStreamer(int inputSize, int outputSize, ClassHandler[] handler) throws IOException {
		DataStreamer ds = new DataStreamer(inputSize, outputSize);
		String[] data;
		while ((data = reader.readNext()) != null) {
			
			float input[] = new float[inputSize];
			float output[] = new float[outputSize];
			for (int i = 0; i < data.length; i++) {
				if (i < inputSize) {
					input[i] = Float.parseFloat(data[i].trim());
				} else {
					output[i - inputSize] = (float) handler[i - inputSize].getClass(data[i].trim());
				}
			}
			
			ds.add(input, output);
		}
		return ds;
	}
}
