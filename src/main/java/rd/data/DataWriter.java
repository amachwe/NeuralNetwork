package rd.data;

import java.util.Map;

/**
 * Data Writer interface
 * @author azahar
 *
 */
public interface DataWriter {


	/**
	 * Default implementation
	 * @param row
	 */
	default void write(Map<String,Object> row)
	{
		System.err.println(row);
	}
	
	/**
	 * Close - must be called to close the data writer, as data writer could be writing to a database, file etc.
	 */
	void close();
}
