package rd.neuron.neuron;

import org.jblas.FloatMatrix;

/**
 * Calculate the Error at the output of the network
 * @author azahar
 *
 */
public class NetworkError {

	private float cumulativeError = 0;
	private int errorCount = 0;
	/**
	 * 
	 * @param error - starting cumulative error
	 */
	public NetworkError(float error)
	{
		this.cumulativeError = error;
	}
	
	public NetworkError()
	{
		
	}
	
	/**
	 * Current error
	 * @param expected
	 * @param actual
	 * @return current error
	 */
	public float currentError(FloatMatrix expected,FloatMatrix actual)
	{
		
		float localError = 0;
		if(expected.isRowVector() && actual.isRowVector() && expected.getRows() == actual.getRows())
		{
			for(int i=0;i<actual.getRows();i++)
			{
				localError += Math.pow(Math.abs(expected.get(i,0)-actual.get(i,0)),2);
			}
			//Cumulative Error
			cumulativeError += localError/2;
			errorCount++;
			return localError/2;
		}
		else
		{
			System.err.println("Error: Bad comparison");
			return Float.MAX_VALUE;
		}
	}
	
	public void reset()
	{
		cumulativeError = 0;
		errorCount = 0;
	}
	/**
	 * Get cumulative error
	 * @return
	 */
	public float getError()
	{
		return cumulativeError;
	}
	
	/**
	 * Get count of error entries
	 * @return
	 */
	public int getErrorCount()
	{
		return errorCount;
	}
}
