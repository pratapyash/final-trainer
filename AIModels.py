import ModelInterfaces
import torch
import numpy as np
import nemo.collections.asr as nemo_asr

from omegaconf import OmegaConf, open_dict


class NeuralASR(ModelInterfaces.IASRModel):
    word_locations_in_samples = None
    audio_transcript = None

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def getTranscript(self) -> str:
        """Get the transcripts of the process audio"""
        assert (
            self.audio_transcript != None,
            "Can get audio transcripts without having processed the audio",
        )
        return self.audio_transcript

    def getWordLocations(self) -> list:
        """Get the pair of words location from audio"""
        assert (
            self.word_locations_in_samples != None,
            "Can get word locations without having processed the audio",
        )

        return self.word_locations_in_samples

    def processAudio(self, audio: str):
        """Process the audio"""
        decoding_cfg = self.model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True
            self.model.change_decoding_strategy(decoding_cfg)

        hypotheses = self.model.transcribe([audio], return_hypotheses=True)
        if type(hypotheses) == tuple and len(hypotheses) == 2:
            hypotheses = hypotheses[0]

        timestamp_dict = hypotheses[0].timestep
        time_stride = 8 * self.model.cfg.preprocessor.window_stride
        word_timestamps = timestamp_dict["word"]

        for stamp in word_timestamps:
            stamp["start_ts"] = stamp.pop("start_offset") * time_stride
            stamp["end_ts"] = stamp.pop("end_offset") * time_stride

        self.word_locations_in_samples = word_timestamps
        self.audio_transcript = hypotheses[0].text


class NeuralTTS(ModelInterfaces.ITextToSpeechModel):
    def __init__(self, model: torch.nn.Module, sampling_rate: int) -> None:
        super().__init__()
        self.model = model
        self.sampling_rate = sampling_rate

    def getAudioFromSentence(self, sentence: str) -> np.array:
        with torch.inference_mode():
            audio_transcript = self.model.apply_tts(
                texts=[sentence], sample_rate=self.sampling_rate
            )[0]

        return audio_transcript


# import ModelInterfaces
# import torch
# import numpy as np


# class NeuralASR(ModelInterfaces.IASRModel):
#     word_locations_in_samples = None
#     audio_transcript = None

#     def __init__(self, model: torch.nn.Module, decoder) -> None:
#         super().__init__()
#         self.model = model
#         self.decoder = decoder  # Decoder from CTC-outputs to transcripts

#     def getTranscript(self) -> str:
#         """Get the transcripts of the process audio"""
#         assert(self.audio_transcript != None,
#                'Can get audio transcripts without having processed the audio')
#         return self.audio_transcript

#     def getWordLocations(self) -> list:
#         """Get the pair of words location from audio"""
#         assert(self.word_locations_in_samples != None,
#                'Can get word locations without having processed the audio')

#         return self.word_locations_in_samples

#     def processAudio(self, audio: torch.Tensor):
#         """Process the audio"""
#         audio_length_in_samples = audio.shape[1]
#         with torch.inference_mode():
#             nn_output = self.model(audio)

#             self.audio_transcript, self.word_locations_in_samples = self.decoder(
#                 nn_output[0, :, :].detach(), audio_length_in_samples, word_align=True)


# class NeuralTTS(ModelInterfaces.ITextToSpeechModel):
#     def __init__(self, model: torch.nn.Module, sampling_rate: int) -> None:
#         super().__init__()
#         self.model = model
#         self.sampling_rate = sampling_rate

#     def getAudioFromSentence(self, sentence: str) -> np.array:
#         with torch.inference_mode():
#             audio_transcript = self.model.apply_tts(texts=[sentence],
#                                                     sample_rate=self.sampling_rate)[0]

#         return audio_transcript


# class NeuralTranslator(ModelInterfaces.ITranslationModel):
#     def __init__(self, model: torch.nn.Module, tokenizer) -> None:
#         super().__init__()
#         self.model = model
#         self.tokenizer = tokenizer

#     def translateSentence(self, sentence: str) -> str:
#         """Get the transcripts of the process audio"""
#         tokenized_text = self.tokenizer(sentence, return_tensors='pt')
#         translation = self.model.generate(**tokenized_text)
#         translated_text = self.tokenizer.batch_decode(
#             translation, skip_special_tokens=True)[0]

#         return translated_text
