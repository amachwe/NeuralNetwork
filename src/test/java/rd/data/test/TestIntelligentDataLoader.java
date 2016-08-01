package rd.data.test;

import static org.junit.Assert.assertNotNull;

import java.io.IOException;

import org.junit.Test;

import rd.data.DefinitionParser;
import rd.data.IntelligentDataLoader;
/**
 * Test Intelligent Data Loader Streaming option
 * @author azahar
 *
 */
public class TestIntelligentDataLoader {

	@Test
	public void doTest() {
		try {
			IntelligentDataLoader loader = new IntelligentDataLoader("d:\\ml stats\\credit\\crx.data", "");
			DefinitionParser dp = new DefinitionParser("d:\\ml stats\\credit\\crx.names");
			loader.stream().forEach(item -> {
				assertNotNull(item);
			});
			assertNotNull(dp.getNumberOfAttributes());
		} catch (IOException e) {

			e.printStackTrace();
		}
	}
}
