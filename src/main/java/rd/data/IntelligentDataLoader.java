package rd.data;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

/**
 * Load CSV data from file
 * 
 * @author azahar
 *
 */
public class IntelligentDataLoader {
	private final String fileName;

	public IntelligentDataLoader(String fileName, String definitionFile) throws FileNotFoundException {
		this.fileName = fileName;
	}

	public Stream<CSVRecord> stream() throws FileNotFoundException {
		Reader in = new FileReader(fileName);

		try {
			return StreamSupport.stream(CSVFormat.DEFAULT.parse(in).spliterator(), false);
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e);
			try {
				in.close();
			} catch (IOException e1) {
				e1.printStackTrace();
				System.err.println(e1);
			}
		}
		return null;
	}

}
