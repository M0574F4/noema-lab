from pathlib import Path
from typing import Iterable, Optional

from noema_lab.core.external_adapters import register_external_adapter_operations
from noema_lab.core.operations import OperationRegistry
from noema_lab.ops.channel.digital import (
    BitBoundaryCheckpointOperation,
    BitCountMatchOperation,
    BitsToIndicesOperation,
    CapacityOracleDigitalLinkOperation,
    CausalCsiPowerAllocatorOperation,
    Crc32CheckOperation,
    Crc32PacketizeOperation,
    Crc32PacketizeV2Operation,
    DigitalDemodulateOperation,
    DigitalModulateOperation,
    NeuralReceiverAdapterOperation,
    OfdmChannelStateOperation,
    OfdmPowerAllocationMetricsOperation,
    IdentityBitLinkOperation,
    IdentityChannelDecoderOperation,
    IdentityChannelEncoderOperation,
    IdentityDemodulateOperation,
    IdentityModulateOperation,
    IdentitySymbolLinkOperation,
    IndicesToBitsOperation,
    PayloadPassthroughDecoderOperation,
    PayloadPassthroughEncoderOperation,
    RepetitionChannelDecoderOperation,
    RepetitionChannelEncoderOperation,
    ReceiverIqImpairmentOperation,
    SymbolBoundaryCheckpointOperation,
    SymbolCountMatchOperation,
    SymbolPowerIdentityOperation,
    SymbolPowerNormalizeOperation,
    SymbolPowerAllocatorOperation,
    WirelessChannelOperation,
    WirelessDigitalLinkOperation,
)
from noema_lab.ops.channel.tensor import BitsToLatentsOperation, LatentsToBitsOperation
from noema_lab.ops.channel.nr_ldpc import (
    CommunicationResourceAccountingOperation,
    CommunicationResourceAccountingV2Operation,
    NrLdpcDecoderOperation,
    NrLdpcEncoderOperation,
)
from noema_lab.ops.metrics.bits import BitErrorRateOperation, BlockErrorRateOperation
from noema_lab.ops.metrics.image import (
    ImageDeliveryStatusOperation,
    ImageReconstructionMetricsOperation,
)
from noema_lab.ops.metrics.task import (
    CaptioningMetricsOperation,
    ClassificationMetricsOperation,
    ClipRetrievalRankOperation,
    DetectionMetricsOperation,
    EmbeddingSimilarityMetricsOperation,
    RetrievalMetricsOperation,
    SegmentationMetricsOperation,
    VqaMetricsOperation,
)
from noema_lab.ops.metrics.text import TextSemanticSimilarityMetricsOperation
from noema_lab.ops.foundation import (
    ClipImageEmbeddingOperation,
    ClipTextEmbeddingOperation,
    DiffusionSemanticStateToImageOperation,
    KnowledgeBaseSourceOperation,
    SamImageSegmentsOperation,
    SemanticStateFaithfulnessMetricsOperation,
    SemanticStateGroundOperation,
    SemanticStatePayloadDecodeOperation,
    SemanticStatePayloadEncodeOperation,
    SemanticStateToTextOperation,
    TextMaskRepairOperation,
    TextSemanticStateEncodeOperation,
    VlmImageSemanticStateOperation,
)
from noema_lab.ops.models.external import (
    DeepJsccExternalDecodeOperation,
    DeepJsccExternalEncodeOperation,
    ExternalBitsDecodeOperation,
    ExternalBitsEncodeOperation,
    ExternalIndicesDecodeOperation,
    ExternalIndicesEncodeOperation,
    ExternalLatentsDecodeOperation,
    ExternalLatentsEncodeOperation,
)
from noema_lab.ops.ai_phy import (
    AiPhyChannelRealizationSourceOperation,
    AiPhyPilotPatternSourceOperation,
    AiPhyPilotChannelSourceOperation,
    AoaEstimationMetricsOperation,
    AoaEstimatorAdapterOperation,
    AoaSceneSourceOperation,
    BeamformingAdapterOperation,
    BeamformingMetricsOperation,
    BeamformingScenarioSourceOperation,
    ChannelEstimationMetricsOperation,
    ChannelEstimatorAdapterOperation,
    EqualPowerAllocationOperation,
    LocalizationAdapterOperation,
    LocalizationGeometrySourceOperation,
    LocalizationMetricsOperation,
    LocalizationScenarioSourceOperation,
    LsChannelEstimatorOperation,
    MrtBeamformerOperation,
    MusicAoaEstimatorOperation,
    PilotObservationOperation,
    RangeObservationOperation,
    ResourceAllocationAdapterOperation,
    ResourceAllocationMetricsOperation,
    ResourceAllocationScenarioSourceOperation,
    TrilaterationLocalizationOperation,
    UlaArrayObservationOperation,
    WaterFillingPowerAllocationOperation,
)
from noema_lab.ops.csi_feedback import (
    CsiFeedbackDecoderOperation,
    CsiFeedbackEncoderOperation,
    CsiFeedbackLinkOperation,
    CsiFeedbackMetricsOperation,
    CsiMrtPrecoderOperation,
    MisoOfdmCsiOperation,
)
from noema_lab.ops.modulation_recognition import (
    ModulationAwgnObservationOperation,
    ModulationClassificationMetricsOperation,
    ModulationClassifierAdapterOperation,
    ModulationFrameSourceOperation,
)
from noema_lab.ops.phase_tracking import (
    CarrierPhaseImpairmentOperation,
    PhaseTrackingReceiverAdapterOperation,
    QpskPilotModulateOperation,
)
from noema_lab.ops.future_wireless import (
    IsacOfdmAllocatorOperation,
    IsacOfdmMetricsOperation,
    IsacOfdmScenarioOperation,
    LeoNtnTrackingAdapterOperation,
    LeoNtnTrackingMetricsOperation,
    LeoNtnTrackingScenarioOperation,
    NearFieldEstimatorAdapterOperation,
    NearFieldMetricsOperation,
    NearFieldXlMimoScenarioOperation,
)
from noema_lab.ops.resource_reliability import (
    NrLdpcOfdmDeliveryMetricsOperation,
    OfdmDelayedCsiOperation,
    OfdmFiniteBlocklengthReliabilityMetricsOperation,
)
from noema_lab.ops.external_task import (
    ExternalClassificationDatasetOperation,
    ExternalClassificationMetricOperation,
)
from noema_lab.ops.models.text_codec import (
    TextBartJsccDecodeOperation,
    TextBartJsccEncodeOperation,
    TextUtf8DecodeOperation,
    TextUtf8EncodeOperation,
)
from noema_lab.ops.models.eflic import (
    EfLicAotInductorDecodeOperation,
    EfLicAotInductorDecodeIndicesOperation,
    EfLicAotInductorEncodeOperation,
    EfLicAotInductorEncodeIndicesOperation,
    EfLicAotInductorExportOperation,
    EfLicBitsToIndicesOperation,
    EfLicDecodeOperation,
    EfLicDecodeIndicesOperation,
    EfLicEncodeOperation,
    EfLicEncodeIndicesOperation,
    EfLicIndicesToBitsOperation,
    EfLicOnnxDecodeOperation,
    EfLicOnnxDecodeIndicesOperation,
    EfLicOnnxEncodeOperation,
    EfLicOnnxEncodeIndicesOperation,
    EfLicOnnxExportOperation,
    EfLicOpenVinoDecodeOperation,
    EfLicOpenVinoDecodeIndicesOperation,
    EfLicOpenVinoEncodeOperation,
    EfLicOpenVinoEncodeIndicesOperation,
)
from noema_lab.ops.models.learned_codecs import (
    CompressAiAnalysisEncodeOperation,
    CompressAiDecodeOperation,
    CompressAiEncodeOperation,
    CompressAiEntropyDecodeOperation,
    CompressAiEntropyEncodeOperation,
    CompressAiAotInductorDecodeOperation,
    CompressAiAotInductorEncodeOperation,
    CompressAiAotInductorEntropyDecodeOperation,
    CompressAiAotInductorEntropyEncodeOperation,
    CompressAiAotInductorExportOperation,
    CompressAiOnnxDecodeOperation,
    CompressAiOnnxCppDecodeOperation,
    CompressAiOnnxCppEncodeOperation,
    CompressAiOnnxCppEntropyDecodeOperation,
    CompressAiOnnxCppEntropyEncodeOperation,
    CompressAiOnnxEncodeOperation,
    CompressAiOnnxEntropyDecodeOperation,
    CompressAiOnnxEntropyEncodeOperation,
    CompressAiOnnxExportOperation,
    CompressAiOpenVinoDecodeOperation,
    CompressAiOpenVinoEncodeOperation,
    CompressAiOpenVinoEntropyDecodeOperation,
    CompressAiOpenVinoEntropyEncodeOperation,
    CompressAiSynthesisDecodeOperation,
    DiffusersAutoencoderKlDecodeOperation,
    DiffusersAutoencoderKlEncodeOperation,
    DiffusersVqModelDecodeOperation,
    DiffusersVqModelEncodeOperation,
    JpegCapacityOracleOperation,
    JpegDecodeOperation,
    JpegEncodeOperation,
)
from noema_lab.ops.models.upstream_lic import (
    EvcDecodeOperation,
    EvcEncodeOperation,
    EvcOnnxDecodeOperation,
    EvcOnnxEncodeOperation,
    EvcOnnxExportOperation,
    HpcmDecodeOperation,
    HpcmEncodeOperation,
    TcmDecodeOperation,
    TcmEncodeOperation,
)
from noema_lab.ops.noise.image import RepresentationLatentNoiseOperation, SourceImagePerturbationOperation
from noema_lab.ops.noise.semantic import RepresentationIndexNoiseOperation
from noema_lab.ops.noise.text import SourceTextPerturbationOperation
from noema_lab.ops.source.image_dataset import ImageDatasetOperation
from noema_lab.ops.source.bit_manifest import BitManifestSourceOperation
from noema_lab.ops.source.random_bits import RandomBitsOperation
from noema_lab.ops.source.coco_yolo import Coco128DetectionOperation, Coco8SegmentationOperation
from noema_lab.ops.source.local_npz import LocalNpzImagesOperation
from noema_lab.ops.source.kodak import KodakFilesOperation
from noema_lab.ops.source.retrieval_flickr8k import Flickr8kRetrievalOperation
from noema_lab.ops.source.retrieval_smoke import RetrievalSmokeOperation
from noema_lab.ops.source.semantic_artifacts import SemanticArtifactsSmokeOperation
from noema_lab.ops.source.task_smoke import TaskLabelsSmokeOperation
from noema_lab.ops.source.text_dataset import TextDatasetOperation
from noema_lab.ops.source.vqa_smoke import VqaSmokeOperation
from noema_lab.ops.source.vqa_manifest import VqaManifestOperation
from noema_lab.ops.source.vqa_small import VqaSmallHfOperation
from noema_lab.ops.vqa_goal import (
    VqaAnswerFromPacketOperation,
    VqaPayloadDecodeOperation,
    VqaPayloadEncodeOperation,
    VqaSemanticSelectOperation,
    VqaTransformersAnswerOperation,
)
from noema_lab.ops.vision_yolo import YoloDetectOperation, YoloSegmentOperation


def build_registry(adapter_paths: Optional[Iterable[Path | str]] = None) -> OperationRegistry:
    registry = OperationRegistry()
    register_builtin_operations(registry)
    register_external_adapter_operations(registry, adapter_paths)
    return registry


def register_builtin_operations(registry: OperationRegistry) -> None:
    registry.register(ImageDatasetOperation())
    registry.register(BitManifestSourceOperation())
    registry.register(RandomBitsOperation())
    registry.register(Coco128DetectionOperation())
    registry.register(Coco8SegmentationOperation())
    registry.register(LocalNpzImagesOperation())
    registry.register(KodakFilesOperation())
    registry.register(Flickr8kRetrievalOperation())
    registry.register(RetrievalSmokeOperation())
    registry.register(TextDatasetOperation())
    registry.register(TaskLabelsSmokeOperation())
    registry.register(SemanticArtifactsSmokeOperation())
    registry.register(AiPhyChannelRealizationSourceOperation())
    registry.register(AiPhyPilotPatternSourceOperation())
    registry.register(PilotObservationOperation())
    registry.register(AiPhyPilotChannelSourceOperation())
    registry.register(LsChannelEstimatorOperation())
    registry.register(ChannelEstimatorAdapterOperation())
    registry.register(ChannelEstimationMetricsOperation())
    registry.register(BeamformingScenarioSourceOperation())
    registry.register(MrtBeamformerOperation())
    registry.register(BeamformingAdapterOperation())
    registry.register(BeamformingMetricsOperation())
    registry.register(LocalizationGeometrySourceOperation())
    registry.register(RangeObservationOperation())
    registry.register(LocalizationScenarioSourceOperation())
    registry.register(TrilaterationLocalizationOperation())
    registry.register(LocalizationAdapterOperation())
    registry.register(LocalizationMetricsOperation())
    registry.register(AoaSceneSourceOperation())
    registry.register(UlaArrayObservationOperation())
    registry.register(MusicAoaEstimatorOperation())
    registry.register(AoaEstimatorAdapterOperation())
    registry.register(AoaEstimationMetricsOperation())
    registry.register(IsacOfdmScenarioOperation())
    registry.register(IsacOfdmAllocatorOperation())
    registry.register(IsacOfdmMetricsOperation())
    registry.register(NearFieldXlMimoScenarioOperation())
    registry.register(NearFieldEstimatorAdapterOperation())
    registry.register(NearFieldMetricsOperation())
    registry.register(LeoNtnTrackingScenarioOperation())
    registry.register(LeoNtnTrackingAdapterOperation())
    registry.register(LeoNtnTrackingMetricsOperation())
    registry.register(ResourceAllocationScenarioSourceOperation())
    registry.register(EqualPowerAllocationOperation())
    registry.register(WaterFillingPowerAllocationOperation())
    registry.register(ResourceAllocationAdapterOperation())
    registry.register(ResourceAllocationMetricsOperation())
    registry.register(ModulationFrameSourceOperation())
    registry.register(ModulationAwgnObservationOperation())
    registry.register(ModulationClassifierAdapterOperation())
    registry.register(ModulationClassificationMetricsOperation())
    registry.register(MisoOfdmCsiOperation())
    registry.register(CsiFeedbackEncoderOperation())
    registry.register(CsiFeedbackLinkOperation())
    registry.register(CsiFeedbackDecoderOperation())
    registry.register(CsiMrtPrecoderOperation())
    registry.register(CsiFeedbackMetricsOperation())
    registry.register(VqaSmokeOperation())
    registry.register(VqaManifestOperation())
    registry.register(VqaSmallHfOperation())
    registry.register(JpegEncodeOperation())
    registry.register(JpegCapacityOracleOperation())
    registry.register(JpegDecodeOperation())
    registry.register(EfLicEncodeOperation())
    registry.register(EfLicDecodeOperation())
    registry.register(EfLicEncodeIndicesOperation())
    registry.register(EfLicDecodeIndicesOperation())
    registry.register(EfLicIndicesToBitsOperation())
    registry.register(EfLicBitsToIndicesOperation())
    registry.register(EfLicOnnxExportOperation())
    registry.register(EfLicAotInductorExportOperation())
    registry.register(EfLicOnnxEncodeOperation())
    registry.register(EfLicOnnxDecodeOperation())
    registry.register(EfLicAotInductorEncodeOperation())
    registry.register(EfLicAotInductorDecodeOperation())
    registry.register(EfLicOpenVinoEncodeOperation())
    registry.register(EfLicOpenVinoDecodeOperation())
    registry.register(EfLicOnnxEncodeIndicesOperation())
    registry.register(EfLicOnnxDecodeIndicesOperation())
    registry.register(EfLicAotInductorEncodeIndicesOperation())
    registry.register(EfLicAotInductorDecodeIndicesOperation())
    registry.register(EfLicOpenVinoEncodeIndicesOperation())
    registry.register(EfLicOpenVinoDecodeIndicesOperation())
    registry.register(CompressAiEncodeOperation())
    registry.register(CompressAiDecodeOperation())
    registry.register(CompressAiAnalysisEncodeOperation())
    registry.register(CompressAiSynthesisDecodeOperation())
    registry.register(CompressAiEntropyEncodeOperation())
    registry.register(CompressAiEntropyDecodeOperation())
    registry.register(CompressAiAotInductorExportOperation())
    registry.register(CompressAiAotInductorEncodeOperation())
    registry.register(CompressAiAotInductorEntropyEncodeOperation())
    registry.register(CompressAiAotInductorEntropyDecodeOperation())
    registry.register(CompressAiAotInductorDecodeOperation())
    registry.register(CompressAiOnnxExportOperation())
    registry.register(CompressAiOnnxEncodeOperation())
    registry.register(CompressAiOnnxEntropyEncodeOperation())
    registry.register(CompressAiOnnxEntropyDecodeOperation())
    registry.register(CompressAiOnnxDecodeOperation())
    registry.register(CompressAiOnnxCppEncodeOperation())
    registry.register(CompressAiOnnxCppEntropyEncodeOperation())
    registry.register(CompressAiOnnxCppEntropyDecodeOperation())
    registry.register(CompressAiOnnxCppDecodeOperation())
    registry.register(CompressAiOpenVinoEncodeOperation())
    registry.register(CompressAiOpenVinoEntropyEncodeOperation())
    registry.register(CompressAiOpenVinoEntropyDecodeOperation())
    registry.register(CompressAiOpenVinoDecodeOperation())
    registry.register(TcmEncodeOperation())
    registry.register(TcmDecodeOperation())
    registry.register(HpcmEncodeOperation())
    registry.register(HpcmDecodeOperation())
    registry.register(EvcEncodeOperation())
    registry.register(EvcDecodeOperation())
    registry.register(EvcOnnxExportOperation())
    registry.register(EvcOnnxEncodeOperation())
    registry.register(EvcOnnxDecodeOperation())
    registry.register(DiffusersAutoencoderKlEncodeOperation())
    registry.register(DiffusersAutoencoderKlDecodeOperation())
    registry.register(DiffusersVqModelEncodeOperation())
    registry.register(DiffusersVqModelDecodeOperation())
    registry.register(ExternalIndicesEncodeOperation())
    registry.register(ExternalIndicesDecodeOperation())
    registry.register(ExternalLatentsEncodeOperation())
    registry.register(ExternalLatentsDecodeOperation())
    registry.register(ExternalBitsEncodeOperation())
    registry.register(ExternalBitsDecodeOperation())
    registry.register(DeepJsccExternalEncodeOperation())
    registry.register(DeepJsccExternalDecodeOperation())
    registry.register(ExternalClassificationDatasetOperation())
    registry.register(ExternalClassificationMetricOperation())
    registry.register(SourceImagePerturbationOperation())
    registry.register(RepresentationIndexNoiseOperation())
    registry.register(RepresentationLatentNoiseOperation())
    registry.register(SourceTextPerturbationOperation())
    registry.register(TextUtf8EncodeOperation())
    registry.register(TextUtf8DecodeOperation())
    registry.register(TextBartJsccEncodeOperation())
    registry.register(TextBartJsccDecodeOperation())
    registry.register(KnowledgeBaseSourceOperation())
    registry.register(TextSemanticStateEncodeOperation())
    registry.register(SemanticStateGroundOperation())
    registry.register(SemanticStatePayloadEncodeOperation())
    registry.register(SemanticStatePayloadDecodeOperation())
    registry.register(SemanticStateToTextOperation())
    registry.register(TextMaskRepairOperation())
    registry.register(ClipTextEmbeddingOperation())
    registry.register(ClipImageEmbeddingOperation())
    registry.register(ClipRetrievalRankOperation())
    registry.register(SamImageSegmentsOperation())
    registry.register(VlmImageSemanticStateOperation())
    registry.register(DiffusionSemanticStateToImageOperation())
    registry.register(VqaSemanticSelectOperation())
    registry.register(VqaPayloadEncodeOperation())
    registry.register(VqaPayloadDecodeOperation())
    registry.register(VqaAnswerFromPacketOperation())
    registry.register(VqaTransformersAnswerOperation())
    registry.register(YoloDetectOperation())
    registry.register(YoloSegmentOperation())
    registry.register(IndicesToBitsOperation())
    registry.register(LatentsToBitsOperation())
    registry.register(BitBoundaryCheckpointOperation())
    registry.register(SymbolBoundaryCheckpointOperation())
    registry.register(SymbolPowerIdentityOperation())
    registry.register(SymbolPowerNormalizeOperation())
    registry.register(OfdmChannelStateOperation())
    registry.register(OfdmDelayedCsiOperation())
    registry.register(SymbolPowerAllocatorOperation())
    registry.register(CausalCsiPowerAllocatorOperation())
    registry.register(OfdmPowerAllocationMetricsOperation())
    registry.register(OfdmFiniteBlocklengthReliabilityMetricsOperation())
    registry.register(NrLdpcOfdmDeliveryMetricsOperation())
    registry.register(PayloadPassthroughEncoderOperation())
    registry.register(PayloadPassthroughDecoderOperation())
    registry.register(IdentityChannelEncoderOperation())
    registry.register(RepetitionChannelEncoderOperation())
    registry.register(NrLdpcEncoderOperation())
    registry.register(IdentityBitLinkOperation())
    registry.register(Crc32PacketizeOperation())
    registry.register(Crc32PacketizeV2Operation())
    registry.register(Crc32CheckOperation())
    registry.register(CapacityOracleDigitalLinkOperation())
    registry.register(IdentitySymbolLinkOperation())
    registry.register(IdentityModulateOperation())
    registry.register(DigitalModulateOperation())
    registry.register(CommunicationResourceAccountingOperation())
    registry.register(CommunicationResourceAccountingV2Operation())
    registry.register(WirelessChannelOperation())
    registry.register(ReceiverIqImpairmentOperation())
    registry.register(QpskPilotModulateOperation())
    registry.register(CarrierPhaseImpairmentOperation())
    registry.register(IdentityDemodulateOperation())
    registry.register(DigitalDemodulateOperation())
    registry.register(NeuralReceiverAdapterOperation())
    registry.register(PhaseTrackingReceiverAdapterOperation())
    registry.register(BitCountMatchOperation())
    registry.register(SymbolCountMatchOperation())
    registry.register(IdentityChannelDecoderOperation())
    registry.register(RepetitionChannelDecoderOperation())
    registry.register(NrLdpcDecoderOperation())
    registry.register(BitsToIndicesOperation())
    registry.register(BitsToLatentsOperation())
    registry.register(WirelessDigitalLinkOperation())
    registry.register(BitErrorRateOperation())
    registry.register(BlockErrorRateOperation())
    registry.register(ImageReconstructionMetricsOperation())
    registry.register(ImageDeliveryStatusOperation())
    registry.register(TextSemanticSimilarityMetricsOperation())
    registry.register(ClassificationMetricsOperation())
    registry.register(VqaMetricsOperation())
    registry.register(DetectionMetricsOperation())
    registry.register(SegmentationMetricsOperation())
    registry.register(CaptioningMetricsOperation())
    registry.register(RetrievalMetricsOperation())
    registry.register(EmbeddingSimilarityMetricsOperation())
    registry.register(SemanticStateFaithfulnessMetricsOperation())
