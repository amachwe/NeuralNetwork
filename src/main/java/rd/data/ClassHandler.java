package rd.data;

import java.util.HashMap;
import java.util.Map;

public class ClassHandler {

	private final Map<Object, Integer> classData = new HashMap<>();

	public ClassHandler() {

	}

	/**
	 * Class conversion from String to integer code
	 * 
	 * @param classId
	 *            - class Id or label
	 * @return
	 */
	public int getClass(String classId) {
		if (classData.containsKey(classId)) {
			return classData.get(classId);
		} else {
			int classCode = classData.size();
			classData.put(classId, classCode);
			return classCode;

		}
	}

	/**
	 * Flatten the class using one hot
	 * 
	 * @param classId
	 *            - class Id or Label
	 * @param width
	 *            - width of the class one hot vector length = number of
	 *            different classes
	 * @return
	 */
	public float[] getFlatClass(String classId, int width) {
		float[] flatClass = new float[width];

		if (classData.containsKey(classId)) {
			flatClass[classData.get(classId)] = 1;

		} else {
			int classCode = classData.size();
			classData.put(classId, classCode);
			flatClass[classCode] = 1;

		}

		return flatClass;
	}
}
