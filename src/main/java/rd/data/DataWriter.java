package rd.data;

import java.util.Map;

public interface DataWriter {


	default void write(Map<String,Object> row)
	{
		System.err.println(row);
	}
	
	void close();
}
