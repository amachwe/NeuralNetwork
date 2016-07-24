package rd.data;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.Reader;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

public class IntelligentDataLoader {
	public IntelligentDataLoader(String fileName, String definitionFile) throws FileNotFoundException {
		Reader in = new FileReader(fileName);

		try {
			Iterable<CSVRecord> records = CSVFormat.DEFAULT.parse(in);
			for (CSVRecord record : records) {
			System.out.println(record);
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e);
		}
	}
}
