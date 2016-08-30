package rd.data;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class MovementData {

	public static enum Keys {

		PreviousA, PreviousB, PreviousTarget, CurrentA, CurrentB, CurrentTarget
	};

	public static enum State {
		Rise(1), Fall(-1), NoChange(0), Unknown;
		private final Integer val;

		private State(int val) {
			this.val = val;
		}

		private State() {
			this.val = 999;
		}

		public int getIntValue() {
			return this.val;
		}
	};

	private final Map<Keys, Number> currVal;

	private final long currentTimestamp, previousTimestamp;

	private final State currATrend, currBTrend;
	// Range of the difference in seconds e.g. 3600 = 60*60 seconds = 1 hr
	private final long range;
	private final boolean invalid;

	public static MovementData build(long currentTimestamp, long previousTimestamp, Map<Keys, Number> currVal) {
		if (currVal == null) {
			return new MovementData(currentTimestamp, previousTimestamp, currVal, -1d, -1d, -1d, -1d, -1d, -1d);
		}

		return new MovementData(currentTimestamp, previousTimestamp, currVal,
				(double) currVal.getOrDefault(Keys.CurrentA, -1), (double) currVal.getOrDefault(Keys.CurrentB, -1),
				(double) currVal.getOrDefault(Keys.CurrentTarget, -1),
				(double) currVal.getOrDefault(Keys.PreviousA, -1), (double) currVal.getOrDefault(Keys.PreviousB, -1),
				(double) currVal.getOrDefault(Keys.PreviousTarget, -1));

	}

	/**
	 * 
	 * @param currentTimestamp
	 * @param previousTimestamp
	 * @param currVal
	 */
	private MovementData(long currentTimestamp, long previousTimestamp, Map<Keys, Number> currVal, double currA,
			double currB, double currTgt, double prevA, double prevB, double prevTgt) {
		this.currentTimestamp = currentTimestamp;
		this.previousTimestamp = previousTimestamp;

		this.range = currentTimestamp - previousTimestamp;
		if (currVal == null) {
			this.currVal = Collections.emptyMap();
			invalid = true;
			currATrend = State.Unknown;
			currBTrend = State.Unknown;
		} else if (currVal.isEmpty()) {
			invalid = true;
			currATrend = State.Unknown;
			currBTrend = State.Unknown;
			this.currVal = currVal;
		} else {
			this.currVal = currVal;
			if (range > 0) {

				if (currA > 0 && prevA > 0 && currB > 0 && prevB > 0) {
					double currATgt = currTgt / currA;
					double prevATgt = prevTgt / prevA;
					currATrend = getState(currATgt, prevATgt);

					double currBTgt = currTgt / currB;
					double prevBTgt = prevTgt / prevB;
					currBTrend = getState(currBTgt, prevBTgt);
					invalid = false;

				} else {
					invalid = true;
					currATrend = State.Unknown;
					currBTrend = State.Unknown;
				}

			} else {
				invalid = true;
				currATrend = State.Unknown;
				currBTrend = State.Unknown;
			}
		}

	}

	public State[] getTrendValues() {
		return new State[] { currATrend, currBTrend };
	}

	private final State getState(double curr, double prev) {
		if (curr > prev) {
			return State.Rise;
		} else if (curr == prev) {
			return State.NoChange;
		} else {
			return State.Fall;
		}
	}

	public boolean isValid() {
		return !invalid;
	}

	public long getRange() {
		return this.range;
	}

	public Map<Keys, Number> getRawData() {
		return Collections.unmodifiableMap(currVal);
	}

	public long getCurrentTimestamp() {
		return this.currentTimestamp;
	}

	public long getPreviousTimestamp() {
		return this.previousTimestamp;
	}

	public String getTrendKey() {

		if (currATrend == currBTrend) {
			return buildKey(currA, currB, currATrend.toString(), currBTrend.toString(), target, "Identical");
		} else if ((currATrend == State.Rise && currBTrend == State.Fall)
				|| (currATrend == State.Fall && currBTrend == State.Rise)) {
			return buildKey(currA, currB, currATrend.toString(), currBTrend.toString(), target, "Opposite");
		}

		return buildKey(currA, currB, currATrend.toString(), currBTrend.toString(), target, "Other");

	}

	private String buildKey(String currA, String currB, String trendA, String trendB, String target, String joint) {
		return (new StringBuilder()).append(currA).append(", ").append(currB).append(", ").append(currATrend)
				.append(", ").append(currBTrend).append(", ").append(joint).append(", ").append(target).toString();
	}

	private static String currA = "", currB = "", target = "";

	public static void setCurrencyKeys(String _currA, String _currB, String _target) {
		currA = _currA;
		currB = _currB;
		target = _target;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("\n\nCurrent TS: ");
		sb.append(this.currentTimestamp);
		sb.append("\tPrevious TS: ");
		sb.append(this.previousTimestamp);
		sb.append("\nRange: ");
		sb.append(this.range);
		sb.append("\tValid: ");
		sb.append(this.isValid());
		sb.append("\nStates:\n A: ");
		sb.append(this.currATrend.toString());
		sb.append("\n B: ");
		sb.append(this.currBTrend.toString());
		sb.append("\nCurrent Values:\n");
		sb.append(currVal);

		return sb.toString();
	}
}
