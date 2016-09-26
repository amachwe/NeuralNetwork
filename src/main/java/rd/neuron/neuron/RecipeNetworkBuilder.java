package rd.neuron.neuron;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.LayerIf.LayerType;

public class RecipeNetworkBuilder {

	private static final Logger logger = LoggerFactory.getLogger(RecipeNetworkBuilder.class);
	public static final String KEY_STOCHASTIC = "STOCHASTIC", KEY_SIMPLE = "SIMPLE", KEY_RANDOM = "RANDOM",
			KEY_FN_SIGMOID = "SIGMOID", KEY_FN_RELU = "RELU";

	private static final FullyRandomLayerBuilder fullyRandomLayerBuilder = new FullyRandomLayerBuilder();

	public static List<LayerIf> build(String recipe) {
		List<LayerIf> network = new ArrayList<>();
		String[] commands = recipe.split("\n");
		if (validate(commands)) {
			process(network, commands);
			return network;
		}

		return network;
	}

	private static boolean validate(String[] commands) {
		if (commands == null || commands.length == 0) {
			logger.error("Networks must have at least one layer");
			return false;
		}

		for (String line : commands) {
			line = line.trim();
			if (line == null || line.isEmpty()) {
				logger.error("There must be no blank lines in the recipe");
				return false;
			}
			if (line.length() < 10 && line.split(" ").length >= 2) {
				logger.error("Bad command formation use: Type <in count> <out count> <function - optional>");
				return false;
			}
		}
		return true;
	}

	private static void process(List<LayerIf> network, String[] commands) {
		int index = 0;
		int total = commands.length;
		for (String command : commands) {
			LayerType type = index == 0 ? LayerType.FIRST_HIDDEN
					: (index == total - 1) ? LayerType.OUTPUT
							: (index == total - 2) ? LayerType.LAST_HIDDEN : LayerType.HIDDEN;
			String[] keys = command.split(" ");

			if (keys.length >= 2) {
				if (keys[0].equalsIgnoreCase(KEY_STOCHASTIC)) {
					createStochasticLayer(network, Integer.parseInt(keys[1]), Integer.parseInt(keys[2]));

				} else if (keys[0].equalsIgnoreCase(KEY_RANDOM)) {
					Function fn = Function.LOGISTIC;
					if (keys.length >= 4) {
						fn = parseFunction(keys[3]);
					}
					createSimpleRandomLayer(network, Integer.parseInt(keys[1]), Integer.parseInt(keys[2]), fn);
				}

			}
			network.get(index).setLayerIdentity(index, type);
			index++;
		}
	}

	private static Function parseFunction(String function) {
		if (function.equalsIgnoreCase(KEY_FN_SIGMOID)) {
			return Function.LOGISTIC;
		} else if (function.equalsIgnoreCase(KEY_FN_RELU)) {
			return Function.ReLU;
		}

		return Function.LOGISTIC;
	}

	private static void createSimpleRandomLayer(List<LayerIf> network, int nIn, int nOut, Function f) {
		network.add(fullyRandomLayerBuilder.build(nIn, nOut, f));
	}

	private static void createStochasticLayer(List<LayerIf> network, int nIn, int nOut) {
		network.add(StochasticLayerBuilder.build(nIn, nOut));
	}
}
