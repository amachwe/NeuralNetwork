package rd.data;

import java.util.Map;

public interface DataWriter {


	default void write(Map<String,Float> row)
	{
		System.err.println(row);
	}
	
	void close();
}
