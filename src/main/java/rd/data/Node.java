package rd.data;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Iterator;

public class Node<T>
{

  public static void main(String...args)
  {
    String sample = "The quick brown fox jumps over the lazy dog";
    String[] words = sample.toLowerCase().split(" ");

    Node<String> prevNode = null,startNode = null;
    for(String word:words)
    {
      if(prevNode!=null)
      {
        Node<String> newNode = new Node<>(word);
        prevNode.add(newNode);
        prevNode = newNode;
      }
      else {
        prevNode = new Node<String>(word);
        startNode = prevNode;
      }
    }

    Node<String> nextNode = null;
    Node<String> node = startNode;
    while(node!=null)
    {
      System.out.println(node+", "+node.getLinkWeights());
      Iterator<Node<String>> itr = node.getLinkWeights().keySet().iterator();
      if(itr.hasNext())
      {
            node=itr.next();
      }
      else
      {
        node = null;
      }

    }

  }
  private final T value;
  private final Map<Node<T>,Integer> linkWeights=new ConcurrentHashMap<>();
  public Node(T value)
  {
    this.value = value;
  }

  @Override
  public String toString()
  {
    return this.value.toString();
  }

  public Map<Node<T>,Integer> getLinkWeights()
  {
    return Collections.unmodifiableMap(linkWeights);
  }

  public void add(Node<T> node)
  {
    Integer val = null;
    if((val =linkWeights.get(node))!=null)
    {
      linkWeights.put(node,val+1);
    }
    else
    {
      linkWeights.put(node,1);
    }
  }

  @Override
  public boolean equals(Object o)
  {
    try
    {
      if(o instanceof Node && ((Node<T>)o).toString().equalsIgnoreCase(this.value.toString()))
      {
        return true;
      }
    }
    catch(Exception e)
    {

    }

    return false;
  }

  @Override
  public int hashCode()
  {
    return this.value.hashCode();
  }


}
