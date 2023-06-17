
using MnistRoomateCompetition;
using MachineLearningLibrary;
using System.Runtime.CompilerServices;
using MnistRoomateCompetition.ToxicAi;

//loading training data
float[][] trainingData = new float[0][];

//load training answers
int[] trainingAnswers = new int[0]; 

//loading testing data
float[][] testingData = new float[0][];

//load testing answers
int[] testingAnswers = new int[0];

Trainer trainer = new(
    trainingData,
    trainingAnswers,
    testingData,
    testingAnswers,
    new MnistLoss()
);

IAgent agent = ToxicMnistCategorizer.InitAgent("");

//TODO: The acceleration and speed thing is based on the change in the AI.
const float gradientFactor = .1f;
float gradientSpeed = .1f;
for(int i = 0; i < 100; i++)
{
    float gradientAcceleration = trainer.Train(agent, gradientSpeed, 100, TrainOption.Minimize);
    gradientSpeed += gradientAcceleration*gradientFactor;
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
