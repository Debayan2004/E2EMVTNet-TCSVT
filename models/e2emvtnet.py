import torch.nn as nn

from models.mcre import Stage1Encoder, SharedMultiScaleFusion
from models.lecv import (
    ViewPairGenerator,
    LearnableEpipolarPrior,
    DifferentiableCorrespondence,
    CrossViewGraphAggregation
)
from models.rmw import MultiViewQualityAssessment
from models.dual_branch import (
    AdaptiveTextProposalNetwork,
    DynamicBoundaryMemory,
    DetectionFeatureAdapter,
    DetectionHead,
    UnifiedBoundaryEvolutionNetwork
)
from models.memory import DynamicEpisodicMemory
from models.recognition import (
    SharedTokenInteraction,
    ProposalFeatureRefinement,
    CrossModalTokenFusion,
    RecognitionEmbedding,
    MemoryGuidedDecoder,
    CharacterPredictionHead,
    RecognitionOutput
)
from models.feedback import (
    BranchPerformanceEstimator,
    AdaptiveFeedbackController,
    TokenWiseAdaptiveFeedback,
    BidirectionalFeatureExchange,
    CollaborativeDisagreementEstimator,
    AdaptiveConvergenceController,
    CollaborativeRefinementCell
)


class E2EMVTNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Stage1Encoder()
        self.fusion = SharedMultiScaleFusion()
        self.pair_generator = ViewPairGenerator()
        self.epipolar = LearnableEpipolarPrior()
        self.correspondence = DifferentiableCorrespondence()
        self.graph = CrossViewGraphAggregation()
        self.quality = MultiViewQualityAssessment()

        self.proposal = AdaptiveTextProposalNetwork()
        self.boundary_memory = DynamicBoundaryMemory()
        self.det_adapter = DetectionFeatureAdapter()
        self.det_head = DetectionHead()
        self.uben = UnifiedBoundaryEvolutionNetwork()

        self.dynamic_memory = DynamicEpisodicMemory()
        self.shared_token = SharedTokenInteraction()
        self.roi_refinement = ProposalFeatureRefinement()
        self.cross_modal = CrossModalTokenFusion()
        self.embedding = RecognitionEmbedding()
        self.decoder = MemoryGuidedDecoder()
        self.character_head = CharacterPredictionHead()
        self.recognition_output = RecognitionOutput()

        self.performance = BranchPerformanceEstimator()
        self.controller = AdaptiveFeedbackController()
        self.token_feedback = TokenWiseAdaptiveFeedback()
        self.exchange = BidirectionalFeatureExchange()
        self.disagreement = CollaborativeDisagreementEstimator()
        self.convergence = AdaptiveConvergenceController()
        self.refinement = CollaborativeRefinementCell()

    def forward(self, images, input_tokens):
        encoder_output = self.encoder(images)
        fusion_output = self.fusion(encoder_output)

        pair_output = self.pair_generator(
            fusion_output["view_features"]
        )

        epipolar_output = self.epipolar(
            pair_output["pair_features"]
        )

        correspondence_output = self.correspondence(
            epipolar_output
        )

        graph_output = self.graph(correspondence_output)
        quality_output = self.quality(graph_output)

        proposal_output = self.proposal(quality_output)

        memory_output = self.boundary_memory(
            proposal_output,
            quality_output
        )

        detection_tokens = self.det_adapter(
            memory_output["memory_tokens"]
        )

        detection_output = self.det_head(detection_tokens)

        boundary_output = self.uben(
            detection_output,
            proposal_output
        )

        recognition_memory = self.dynamic_memory(
            proposal_output,
            quality_output
        )

        shared_output = self.shared_token(
            recognition_memory["shared_memory_tokens"]
        )

        roi_output = self.roi_refinement(
            recognition_memory["projected_tokens"]
        )

        fusion_tokens = self.cross_modal(
            shared_output["shared_tokens"],
            roi_output
        )

        embedding_output = self.embedding(
            input_tokens,
            fusion_tokens
        )

        decoder_output = self.decoder(embedding_output)

        prediction_output = self.character_head(decoder_output)

        recognition_output = self.recognition_output(
            decoder_output,
            prediction_output
        )

        performance_output = self.performance(
            detection_output,
            recognition_output
        )

        controller_output = self.controller(performance_output)

        feedback_output = self.token_feedback(
            detection_output,
            recognition_output,
            controller_output
        )

        exchange_output = self.exchange(feedback_output)

        disagreement_output = self.disagreement(exchange_output)

        convergence_output = self.convergence(
            performance_output,
            controller_output,
            disagreement_output
        )

        refinement_output = self.refinement(
            exchange_output,
            convergence_output
        )

        return {
            "fusion_output": fusion_output,
            "correspondence_output": correspondence_output,
            "graph_output": graph_output,
            "quality_output": quality_output,
            "proposal_output": proposal_output,
            "memory_output": memory_output,
            "detection_output": detection_output,
            "boundary_output": boundary_output,
            "recognition_memory": recognition_memory,
            "recognition_output": recognition_output,
            "performance_output": performance_output,
            "controller_output": controller_output,
            "feedback_output": feedback_output,
            "exchange_output": exchange_output,
            "disagreement_output": disagreement_output,
            "convergence_output": convergence_output,
            "refinement_output": refinement_output
        }