package rd.neuron.neuron;

import java.util.LinkedHashMap;
import java.util.Map;

import org.jblas.FloatMatrix;

import rd.data.DataStreamer;
import rd.data.DataWriter;
/**
 * Simple Network Performance Evaluator
 * @author azahar
 *
 */
public class SimpleNetworkPerformanceEvaluator {

	/**
	 * Keys for Key-Value pairs
	 * @author azahar
	 *
	 */
	public static enum Keys {
		Weight, Bias, RowId, Error, LearningRate, BatchId
	};

	private final DataWriter dw;

	private float rowId = 0;

	public SimpleNetworkPerformanceEvaluator(DataWriter writer) {
		dw = writer;

	}
	
	/**
	 * Evaluate Network Weights
	 * @param sn - Simple Network
	 * @param learningRate - constant learning rate (0.05 - 0.01)
	 */
	public void evaluateNetwork(SimpleNetwork sn, float learningRate)
	{
		Map<String,Object> row = new LinkedHashMap<>();
		row.put(Keys.LearningRate.toString(), learningRate);
		row.put(Keys.RowId.toString(), rowId++);
		
	
		
		int layerCount = 0;
		for (Layer l : sn) {
			int weightCount = 0;
			int neuronCount = 0;
			for (Float weight : l.getWeights().elementsAsList()) {
				row.put(Keys.Weight.toString() + "_" + (layerCount) + "_" + (++weightCount), weight);

			}
			for (Float bias : l.getAllBias().elementsAsList()) {
				row.put(Keys.Bias.toString() + "_" + (layerCount) + "_" + (++neuronCount), bias);

			}
			
			layerCount++;
		}
	
		dw.write(row);

	}

	/**
	 * Evaluate Error and Network Weights
	 * @param sn - Simple Network
	 * @param ds - Data Streamer
	 * @param learningRate - constant learning rate (0.05 - 0.01)
	 */
	public void evaluateErrorAndNetwork(SimpleNetwork sn, DataStreamer ds, float learningRate) {

		// Calc Error
		float error = 0.0f;

		for (FloatMatrix item : ds) {
			FloatMatrix actualOutput = sn.io(item);
			FloatMatrix delta = actualOutput.sub(ds.getOutput(item));
			delta = delta.mul(delta);
			error += Math.sqrt(delta.sum());
		}
		
		Map<String, Object> row = new LinkedHashMap<>();
		row.put(Keys.LearningRate.toString(), learningRate);
		row.put(Keys.RowId.toString(), rowId++);
		
	
		
		int layerCount = 0;
		for (Layer l : sn) {
			int weightCount = 0;
			int neuronCount = 0;
			for (Float weight : l.getWeights().elementsAsList()) {
				row.put(Keys.Weight.toString() + "_" + (layerCount) + "_" + (++weightCount), weight);

			}
			for (Float bias : l.getAllBias().elementsAsList()) {
				row.put(Keys.Bias.toString() + "_" + (layerCount) + "_" + (++neuronCount), bias);

			}
			
			layerCount++;
		}
		row.put(Keys.Error.toString(), error);
		dw.write(row);

	}

}
