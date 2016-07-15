package rd.neuron.neuron;

import org.jblas.FloatMatrix;

public class NetworkError {

	private float error = 0;
	private int errorCount = 0;
	public NetworkError(float error)
	{
		this.error = error;
	}
	
	public NetworkError()
	{
		
	}
	
	public float localError(FloatMatrix expected,FloatMatrix actual)
	{
		
		float localError = 0;
		if(expected.isRowVector() && actual.isRowVector() && expected.getRows() == actual.getRows())
		{
			for(int i=0;i<actual.getRows();i++)
			{
				localError += Math.pow(Math.abs(expected.get(i,0)-actual.get(i,0)),2);
			}
			error += localError/2;
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
		error = 0;
		errorCount = 0;
	}
	public float getError()
	{
		return error;
	}
	
	public int getErrorCount()
	{
		return errorCount;
	}
}
