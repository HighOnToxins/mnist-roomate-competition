﻿
using MachineLearningLibrary;
using System.IO;

namespace MnistRoomateCompetition.ToxicAi;

public class ToxicMnistCategorizer: IMnistRecogniser
{
    public IAgent mnistAgent;

    private const int kernalSize = 3;
    private const int randomStartRange = 10;

    private const int convolutionCount = (Image.Rows - 4) / 2;
    private const int smallImage = Image.Rows - convolutionCount;

    public ToxicMnistCategorizer(string path)
    {
        mnistAgent = InitAgent(path);
    }

    public static IAgent InitAgent(string path)
    {
        try
        {
            return IAgent.LoadFromFile(path);
        }
        catch
        {
            IAgent[] layers = new IAgent[convolutionCount + 1];

            for(int i = 0; i < convolutionCount; i++)
            {
                int inputSize = Image.Rows - 1 - i * 2;
                int outputSize = inputSize - 2;
                layers[i] = new Convolution2DAgent(
                    kernalSize, kernalSize,
                    inputSize, inputSize,
                    outputSize, outputSize,
                    -randomStartRange, randomStartRange
                );
            }

            layers[^1] = new AffineAgent(
                smallImage, 10,
                -randomStartRange, randomStartRange
            );

            return new AgentComposite(layers);
        }
    }

    public Result Test(Image image)
    {
        //TODO: Make Library more flexible in order to handle generic objects
        float[] data = new float[Image.Rows*Image.Rows];
        for(int x = 0; x < Image.Rows; x++)
        {
            for(int y = 0; y < Image.Columns; y++)
            {
                data[x * Image.Rows + y] = image[x, y];
            }
        }

        mnistAgent.Invoke(data, out IReadOnlyList<float> result);
        
        return new Result(result.ToArray());
    }

}
