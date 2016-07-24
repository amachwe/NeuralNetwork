package rd.data.test;

import java.io.FileNotFoundException;
import java.io.IOException;

import rd.data.DefinitionParser;
import rd.data.IntelligentDataLoader;

public class TestIntelligentDataLoader {

	public static void main(String...args)
	{
		try {
			IntelligentDataLoader loader =new IntelligentDataLoader("d:\\ml stats\\credit\\crx.data","");
			DefinitionParser dp = new DefinitionParser("d:\\ml stats\\credit\\crx.names");
		} catch (IOException e) {
			
			e.printStackTrace();
		}
	}
}
