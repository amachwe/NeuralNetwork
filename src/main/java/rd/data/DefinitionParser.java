package rd.data;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class DefinitionParser {

	public static final String CONTINUOUS = "continuous.", CLASS_ATTRIBUTE = "(class attribute)";
	public static final String KEY_TITLE = "Title:", KEY_SOURCES = "Sources:", KEY_PAST_USAGE = "Past Usage:",
			KEY_REL_INFO = "Relevant Information:", KEY_NO_OF_INSTANCES = "Number of Instances:",
			KEY_NO_OF_ATTRIBUTES = "Number of Attributes:", KEY_ATTRIB_INFO = "Attribute Information:",
			KEY_ATTRIB_MISSING = "Missing Attribute Values:", KEY_CLASS_DISTR = "Class Distribution",
			KEY_UNKNOWN = "Unknown";

	public static enum Section {
		TITLE(KEY_TITLE), SOURCES(KEY_SOURCES), NO_OF_ATTRIBUTES(KEY_NO_OF_ATTRIBUTES), NO_OF_INSTANCES(
				KEY_NO_OF_INSTANCES), ATTRIBUTE_INFO(KEY_ATTRIB_INFO), ATTRIBUTE_MISSING(
						KEY_ATTRIB_MISSING), CLASS_DISTR(KEY_CLASS_DISTR), UNKNOWN(
								KEY_UNKNOWN), PAST_USAGE(KEY_PAST_USAGE), REL_INFO(KEY_REL_INFO);
		private final String key;

		private Section(String key) {
			this.key = key;
		}

		public final String getKey() {
			return key;
		}
	};

	private final List<String> headers = new ArrayList<>();

	public DefinitionParser(String fileName) throws IOException {

		Files.lines(FileSystems.getDefault().getPath(fileName)).forEach(line -> {
			line = line.trim();
			if (!line.isEmpty()) {

				parse(line);
			}
		});
		System.out.println(sectionMap);
	}

	private int lineCount = 0;
	private int noOfAttribs = 0, noOfInstances = 0;
	private String noOfAttribs_Desc = "", title = "";
	private Section currentSection = Section.UNKNOWN;

	private Map<Section, String> sectionMap = new LinkedHashMap<>();
	private String sectionText = "";

	private void parse(String text) {
		System.out.println((++lineCount) + "] " + currentSection + " [" + text);

		if (text.contains(KEY_TITLE)) {

			updateSectionMap(sectionMap, currentSection, sectionText);
			currentSection = Section.TITLE;
			sectionText = "";
		}

		if (text.contains(KEY_NO_OF_ATTRIBUTES)) {

			updateSectionMap(sectionMap, currentSection, sectionText);
			currentSection = Section.NO_OF_ATTRIBUTES;
			sectionText = "";

		}

		if (text.contains(KEY_NO_OF_INSTANCES)) {
			updateSectionMap(sectionMap, currentSection, sectionText);
			currentSection = Section.NO_OF_INSTANCES;
			sectionText = "";

		}
		if (text.contains(KEY_PAST_USAGE)) {
			updateSectionMap(sectionMap, currentSection, sectionText);
			currentSection = Section.PAST_USAGE;
			sectionText = "";
		}
		if (text.contains(KEY_REL_INFO)) {
			updateSectionMap(sectionMap, currentSection, sectionText);
			currentSection = Section.REL_INFO;
			sectionText = "";
		}
		if (text.contains(KEY_SOURCES)) {
			updateSectionMap(sectionMap, currentSection, sectionText);
			currentSection = Section.SOURCES;
			sectionText = "";
		}
		if (text.contains(KEY_ATTRIB_MISSING)) {
			updateSectionMap(sectionMap, currentSection, sectionText);
			currentSection = Section.ATTRIBUTE_MISSING;
			sectionText = "";
		}
		if (text.contains(KEY_CLASS_DISTR)) {
			updateSectionMap(sectionMap, currentSection, sectionText);
			currentSection = Section.CLASS_DISTR;
		}

		if (text.contains(KEY_ATTRIB_INFO)) {
			updateSectionMap(sectionMap, currentSection, sectionText);
			currentSection = Section.ATTRIBUTE_INFO;
			sectionText = "";
		}
		sectionText += text + "\n";
	}

	private void updateSectionMap(Map<Section, String> _sectionMap, Section _currentSection, String _sectionText) {
		if (_currentSection != Section.UNKNOWN) {
			_sectionMap.put(_currentSection, getValue(_currentSection.getKey(), _sectionText));
		} else {
			System.err.println("Error: Map not updated for Unknown Section (Section Text): " + _sectionText);
		}
	}

	public List<String> getHeaders() {
		return headers;
	}

	public String getNumberOfAttributesDescription() {
		return noOfAttribs_Desc;
	}

	public int getNumberOfAttributes() {
		return noOfAttribs;
	}

	public int getNumberOfInstances() {
		return noOfInstances;
	}

	private String getValue(String key, String text) {
		return text.substring(text.indexOf(key) + key.length() + 1);
	}

}
