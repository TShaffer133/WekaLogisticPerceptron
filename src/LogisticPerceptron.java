import java.lang.Math;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities;

public class LogisticPerceptron implements weka.classifiers.Classifier
{
  private String fileName;
  private String numberOfEpochs;
  private String learningRate;
  private String lambdaSquash;
  private int weightUpdates = 0;
  private double[] weights;
  private double[] errors;

  //Instantiate and set values
  public LogisticPerceptron(String[] options)
  {
    this.fileName = options[0];
    this.numberOfEpochs = options[1];
    this.learningRate = options[2];
    this.lambdaSquash = options[3];
  }

  //Implementation of logistic perceptron
  @Override
  public void buildClassifier(Instances data) throws Exception
  {
    //Print out header
    System.out.println("University of Central Florida\nCAP4630 Artificial Intelligence - Fall 2017\nLogistic Perceptron Classifier by Tyrone Shaffer\n");

    //Create weights array that includes bias
    weights = new double[data.numAttributes()];

    //Array to store instance error values to calculate mean squared error
    errors = new double[data.numInstances()];

    //Begin modeling using the Perceptron algorithm
    for (int i = 0; i < Integer.parseInt(numberOfEpochs); i++)
    {
      System.out.print("Epoch " + i +":\t");

      for(int j = 0; j < data.numInstances(); j++)
      {
        //Get values for current instance
        Instance currentInst = data.instance(j);
        double [] input = currentInst.toDoubleArray();
        //Convert class value into bias for input array
        input[input.length-1] = 1;

        //Get net value for current instance
        double sum = calculateSum(input);
        double threshold = logisticThreshold(sum);
        int predict = predict(currentInst);
        
        if(predict < 0)
            predict = 0;

        //Only calculate delta and adjust weights if predicted != actual
        if(predict != currentInst.classValue())
        {
          weightUpdates++;
          
          double error = calculateError(currentInst, threshold);
          double delta = calculateDelta(sum, error);
          //System.out.println("Delta: " +delta);
          adjustWeights(delta, input, error);
          
          /*
          for (double d : weights)
          {
              System.out.println(d);
          }
          */
          
          System.out.print("0");
          
        }
        else
          System.out.print("1");
        
        
        System.out.print(": actualValue: " + currentInst.classValue());
        System.out.print(" Predicted: " + predict);
        System.out.print(" Threshold: " + threshold);
        System.out.println(" Sum: " + sum);
        
      }
      
      //Separate line for next epoch
      System.out.print("\n");
    }

  }

  //Helper function to calculate "net"
  private double calculateSum(double [] input)
  {
    double sum = 0.0;

    //Add in all of the instance data and weights
    for(int i = 0; i < input.length; i++)
    {
      sum += (input[i] * weights[i]);
    }

    return sum;
  }

  //Helper function to calculate the threshold function
  private double logisticThreshold(double sum)
  {
    double e = Math.exp((-1*Double.parseDouble(lambdaSquash))*sum);
    double result = 1/(1+e);
    return result;
  }

  private double calculateError(Instance data, double prediction)
  {
    double actualValue = data.classValue();
    if(actualValue == 0)
        actualValue = -1; 
    else
        prediction = -prediction;
    
    double result = actualValue - prediction;
    return result;
  }

  //Helper function to calculate delta value based on gradient descent
  private double calculateDelta(double sum, double error)
  {
    //All these commands can be integrated into one line.
    //Just want to ensure no funny business.
    double e = Math.exp((-1*Double.parseDouble(lambdaSquash))*sum);
    double devisor = Math.pow((1+e), 2.0);
    double numerator = Double.parseDouble(lambdaSquash)*e;
    double logisticDerivative = numerator/devisor;

    double result = Double.parseDouble(learningRate) * error * logisticDerivative;
    return result;
  }

  //Helper function to adjust weights based on delta
  private void adjustWeights(double delta, double[] input, double error)
  {
    //System.out.println(delta);
    for(int i = 0; i < input.length - 1; i++)
    {
      weights[i] = weights[i] + (delta * input[i]);
    }
    weights[weights.length-1] += (error * input[input.length-1] * Double.parseDouble(learningRate));
  }

  //Helper function to calculate mean squared error
  private double calculateMeanSquaredError(double[] errors)
  {
    double sum = 0;
    for(int i = 0; i < errors.length; i++)
    {
      double squared = Math.pow(errors[i], 2.0);
      sum += squared;
    }

    return (sum/2);
  }

  //Required by classifier interface, does not need to be filled
  @Override
  public Capabilities getCapabilities()
  {
    return null;
  }

  //Required by classifier interface, does not need to be filled
  @Override
  public double classifyInstance(Instance instance)
  {
    return 0.0;
  }

  //Output results, required by LogisticWeka
  @Override
  public String toString()
  {
    String allTheThings = new String("Source file\t: " + this.fileName + "\n"
                                  +"Training epochs\t : " + this.numberOfEpochs +"\n"
                                  +"Learning rate\t : " + this.learningRate + "\n"
                                  +"Lambda value\t : " + this.lambdaSquash + "\n"
                                  +"Total # weight updates = " + this.weightUpdates +"\n"
                                  +"Final weights:" +"\n");

    for(int i = 0; i < weights.length; i++)
    {
      allTheThings = allTheThings.concat(weights[i] + "\n");
    }

    return allTheThings;
  }

  //Required by Weka, Credit: Dr. Demetrios Glinos
  @Override
  public double[] distributionForInstance(Instance instance)
  {
    double[] result = new double[2];
    if(predict(instance) < 0)
    {
      result[0] = 1;
      result[1] = 0;
    }
    else
    {
      result[0] = 0;
      result[1] = 1;
    }
    return result;
  }

  //Helper function, outputs predicted value based on threshold
  private int predict(Instance instance)
  {
      double [] input = instance.toDoubleArray();
      input[input.length-1] = 1;
      double sum = calculateSum(input);
   
      if(sum <= 0)
        return -1;
      else
        return 1;

  }
}
