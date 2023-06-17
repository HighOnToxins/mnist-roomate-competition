
using MnistRoomateCompetition;
using MachineLearningLibrary;
using System.Runtime.CompilerServices;
using MnistRoomateCompetition.ToxicAi;

//loading training data & training answers
TrainingData[] train = MnistLoader.LoadData("", "");
ConvertToReadOnly(train, out float[][] trainingData, out int[] trainingAnswers);

TrainingData[] test = MnistLoader.LoadData("", "");
ConvertToReadOnly(test, out float[][] testingData, out int[] testingAnswers);

Trainer trainer = new(
    trainingData,
    trainingAnswers,
    testingData,
    testingAnswers,
    new MnistLoss()
);

string agentPath = "../../../ToxicAgent/agent.bin";

IAgent agent = ToxicMnistCategorizer.InitAgent(agentPath);

//TODO: The acceleration and speed thing is based on the change in the AI.
const float gradientFactor = .1f;
float gradientSpeed = .1f;
for(int i = 0; i < 100; i++)
{
    float gradientAcceleration = trainer.Train(agent, gradientSpeed, 100, TrainOption.Minimize);
    gradientSpeed += gradientAcceleration*gradientFactor;
}

IAgent.SaveToFile(agent, agentPath);

void ConvertToReadOnly(TrainingData[] dataData, out float[][] data, out int[] answers)
{
    data = new float[dataData.Length][];
    answers = new int[dataData.Length];
    for(int i = 0; i < dataData.Length; i++)
    {
        data[i] = new float[Image.Rows * Image.Columns];

        for(int x = 0; x < Image.Columns; x++)
        {
            for(int y = 0; y < Image.Rows; y++)
            {
                data[i][x * Image.Rows + y] = train[i].Image[x, y] / 255f;
                answers[i] = train[i].Label;
            }
        }
    }
}

internal static class Util
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe int ToInt(this bool b) => *(byte*)&b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe int ToSign(this bool b) => b.ToInt() * 2 - 1;
}

//mnist loss function
internal sealed class MnistLoss: ILossFunc
{
    public int Answer { get; set; }

    public void Invoke(in IReadOnlyList<float> value, out float result)
    {
        result = 0;
        for(int i = 0; i < value.Count; i++)
        {
            float x = (i == Answer).ToInt() - value[i];
            result += x * (x >= 0).ToSign();
        }
    }

    public void Invoke(in IReadOnlyList<float> value, in IReadOnlyList<float>? gradient, out float valueOutput, out float derivativeResult, int varIndex = -1)
    {
        Invoke(value, out valueOutput);

        derivativeResult = 0;
        
        if(gradient is null)
        {
            derivativeResult = 0;
            return;
        }

        for(int i = 0; i < value.Count; i++)
        {
            float x = (i == Answer).ToInt() - value[i];
            derivativeResult += gradient[i] * (x >= 0).ToSign();
        }
    }
}
