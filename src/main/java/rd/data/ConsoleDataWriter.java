package rd.data;

import java.util.Map;
/**
 * Console Data Writer - write data to the error stream
 * @author azahar
 *
 */
public class ConsoleDataWriter implements DataWriter {

	public void write(Map<String,Object> row)
	{
		String rowText = row.toString();
		System.err.println(rowText.substring(1, rowText.length()-1));
	}
	
	@Override
	public void close() {
		
		
	}

}
